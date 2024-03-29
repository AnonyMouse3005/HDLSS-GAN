"""CTGAN module."""

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional, Sigmoid, BCELoss, CrossEntropyLoss
from tqdm import tqdm
import optuna

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from ctgan.wpfs import FirstLinearLayer


class Classifier(Module):
    """Additional auxiliary classifier for the CTGAN."""

    def __init__(self, trial, input_dim, classifier_dim, st_ed, first_layer):
        super(Classifier, self).__init__()

        self.first_layer = first_layer

        if self.first_layer is not None:
            dim = classifier_dim[0]
        else:
            dim = input_dim - (st_ed[1]-st_ed[0])  # number of transformed features (without targets)
        seq = []
        self.st_ed = st_ed
        for item in list(classifier_dim):  # for each hidden layer
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item
        
        # specify output layer
        if (st_ed[1]-st_ed[0]) == 1:  # regression, DEPRECATED
            seq += [Linear(dim, 1)]
        
        elif (st_ed[1]-st_ed[0]) == 2:  # binary classification
            seq += [Linear(dim, 1), Sigmoid()]
        else:  # multiclass classification
            seq += [Linear(dim,(st_ed[1]-st_ed[0]))] 

        self.seq = Sequential(*seq)

    def forward(self, _input):
        if (self.st_ed[1]-self.st_ed[0]) == 1:  # regression, DEPRECATED
            label = _input[:, self.st_ed[0]:self.st_ed[1]]
        else:  # classification
            label = torch.argmax(_input[:, self.st_ed[0]:self.st_ed[1]], axis=-1)
        
        new_input_ = torch.cat((_input[:, :self.st_ed[0]], _input[:, self.st_ed[1]:]), 1)  # input without target columns

        sparsity_weights = None
        if self.first_layer is not None:
            new_input_, sparsity_weights = self.first_layer(new_input_)

        if (self.st_ed[1]-self.st_ed[0]) <= 2:
            return self.seq(new_input_).view(-1), label, sparsity_weights
        else:
            return self.seq(new_input_), label, sparsity_weights


def get_st_ed(output_info_list):
    """Return (st, ed), total numbers of transformed features w/o and with target variable, respectively (ed = data_dim)."""
    st = 0

    for _, item in output_info_list[:-1]:  # for each raw feature (continuous or discrete) except last one, which is the target
        if len(item) > 1:  # continuous
            st += item[0][0] + item[1][0]
        else:  # discrete
            st += item[0][0]

    ed = st + item[-1][0]

    return (st, ed)


class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, trial, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)  # (#pacs, 1, 1); 'pac' samples in each pac, batch//pac = #pacs
        alpha = alpha.repeat(1, pac, real_data.size(1))  # (#pacs, pac, m); m = 'input_dim' i.e., #cols of real_data
        alpha = alpha.view(-1, real_data.size(1))  # (pac*#pacs, m) = (batch, m); same value in each row for every #pacs row 

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)  # (pac*#pacs, m)

        disc_interpolates = self(interpolates)  # i.e., discriminator(interpolates); (#pacs, 1) i.e., one prediction for each pac

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),  # 'ones' as we want to achieve 1-Lipschitz condition
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]  # -> Tensor, (pac*#pacs, m)

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1  # ||(#pacs, pac*m)||_2 = (#pacs, )
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_  # scalar

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, trial, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


class CTGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256). NOTE: 2 hidden layers, each having 256 neurons
        classifier_dim: same as discriminator_dim
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        classifier_lr (float):
            Learning rate for the classifier. Defaults to 2e-4.
        classifier_decay (float):
            Classifier weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 100.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 3.
        q (int):
            Factor for multiplying to batch size when sampling conditional vectors.
            Defaults to 20.
        omega (float):
            Ratio of conditions in condvec dedicated for the target variable.
            Defaults to 0.5.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256), classifier_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=1e-6, classifier_lr=2e-4, classifier_decay=1e-6,
                 batch_size=30, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=100, pac=3, cuda=True,
                 q=20, sk=False, wpfs=False, wpn_layers=(256, 256, 256, 256), smart_cond=False, omega=0.5):

        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim
        self._classifier_dim = classifier_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay
        self._classifier_lr = classifier_lr
        self._classifier_decay = classifier_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac
        self.q = q
        self.sk = sk
        self.wpfs = wpfs
        if self.wpfs:
            self.wpn_layers = wpn_layers
        self.smart_cond = smart_cond
        self.omega = omega

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for _, column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):  # 'data' before activation
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for _, column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]  # average over all samples in batch

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    def _partition_data(self, real_data, target, test_data=None, synthetic_data=None):
        y_train_real = real_data[target].to_numpy()
        X_train_real = real_data.drop(target, axis=1)
        if test_data is not None:
            y_test = test_data[target].to_numpy()
            X_test = test_data.drop(target, axis=1)
            if synthetic_data is not None:
                y_train_fake = synthetic_data[target].to_numpy()
                X_train_fake = synthetic_data.drop(target, axis=1)
                return X_train_real, X_train_fake, X_test, y_train_real, y_train_fake, y_test
            else:
                return X_train_real, X_test, y_train_real, y_test
        else:
            return X_train_real, y_train_real

    def _get_score(self, clf, X_train, X_test, y_train, y_test):
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:,1]

        auroc = roc_auc_score(y_test, y_pred_proba)
        return auroc

    def evaluate_compatibility(self, clf, real_data, synthetic_data, test_data, target):
        X_train_real, X_train_fake, X_test, y_train_real, y_train_fake, y_test = self._partition_data(real_data, target, test_data, synthetic_data)
        auroc_real = self._get_score(clf, X_train_real, X_test, y_train_real, y_test)
        auroc_fake = self._get_score(clf, X_train_fake, X_test, y_train_fake, y_test)

        return -np.abs(auroc_real-auroc_fake)

    def evaluate_downstream_pred(self, clf, real_data, synthetic_data, test_data, target):
        X_train_real, X_train_fake, X_test, y_train_real, y_train_fake, y_test = self._partition_data(real_data, target, test_data, synthetic_data)
        X_train = pd.concat([X_train_real, X_train_fake], axis=0)
        y_train = np.concatenate((y_train_real, y_train_fake))

        auroc = self._get_score(clf, X_train, X_test, y_train, y_test)
        return auroc

    @random_state
    def fit(self, trial, train_data, discrete_columns=(), epochs=None, task=None, return_loss=False):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
            task (dict):
                Dictionary containing {problem_type: target_column},
                where problem_type in ['Classification', ]
        """
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        if task:
            target = task[list(task)[0]]
            target_idx = train_data.columns.get_loc(target)

        raw_data = train_data.copy()
        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data, transformed_column_names = self._transformer.transform(train_data)
        discrete_col_idxs, transformed_discrete_columns = [], []
        for idx, c in enumerate(transformed_column_names):
            if c.startswith(tuple(discrete_columns)):
                transformed_discrete_columns.append(c)
                discrete_col_idxs.append(idx)

        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency,
            transformed_discrete_columns)

        data_dim = self._transformer.output_dimensions  # number of transformed columns (only beta for continuous columns)

        self._generator = Generator(
            trial,
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim
        ).to(self._device)

        discriminator = Discriminator(
            trial,
            data_dim + self._data_sampler.dim_cond_vec(),  # dim_cond_vec equal data_dim without counting continuous features (excluding both alpha and beta)
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        if task:
            st_ed = get_st_ed(self._transformer.output_info_list)
            discrete_feature_idxs = [col_idx for col_idx in discrete_col_idxs if col_idx not in range(st_ed[0], st_ed[1])]
            if self.wpfs:
                unlabeled_train_data = np.concatenate((train_data[:, :st_ed[0]], train_data[:, st_ed[1]:]), axis=1)
                embedding_matrix = torch.tensor(np.rot90(unlabeled_train_data)[:, :self._embedding_dim].copy(), dtype=torch.float32, requires_grad=False)
                assert self._classifier_dim[0] == self.wpn_layers[-1], "The output size of WPN must be the same as the first layer of the AC."
                first_layer = FirstLinearLayer(
                    self.wpn_layers,
                    self._classifier_dim,
                    embedding_matrix
                )
                other_cond_size = round((1-self.omega)*self._batch_size)
                self.col_idxs = np.random.choice(np.arange(len(transformed_discrete_columns)), other_cond_size) if self.smart_cond else None
            else:
                first_layer = None
                self.col_idxs = None
            classifier = Classifier(
                trial,
                data_dim,
                self._classifier_dim,
                st_ed,
                first_layer
            ).to(self._device)

            optimizerC = optim.Adam(
                classifier.parameters(), lr=self._classifier_lr,
                betas=(0.5, 0.9), weight_decay=self._classifier_decay
            )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):

                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size, sk=self.sk, col_idxs=self.col_idxs)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(train_data, self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt, _ = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                mean_padded = torch.zeros(self._batch_size*self.q, self._embedding_dim, device=self._device)
                std_padded = mean_padded + 1
                fakez = torch.normal(mean=mean_padded, std=std_padded)
                condvec = self._data_sampler.sample_condvec(self._batch_size*self.q, q=self.q, sk=self.sk, col_idxs=self.col_idxs)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt, cond_target = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)  # fake.size() == fakeact.size()

                if c1 is not None:
                    y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()

                if task:
                    fake = self._generator(fakez[cond_target, :])
                    fakeact = self._apply_activate(fake)

                    real_pred, real_label, sparsity_weights = classifier(real)
                    fake_pred, fake_label, _ = classifier(fakeact)
                    if sparsity_weights is not None:
                        importances = sparsity_weights[discrete_feature_idxs]
                        col_idxs = np.random.choice(np.arange(importances.size(dim=0)), size=other_cond_size, p=importances/importances.sum())
                        if self.smart_cond:
                            self.col_idxs = col_idxs.numpy()

                    if (st_ed[1]-st_ed[0]) > 2:  # multiclass classification
                        c_loss = CrossEntropyLoss()
                    elif (st_ed[1]-st_ed[0]) == 2:  # binary classification
                        c_loss = BCELoss()
                        real_label = real_label.type_as(real_pred)
                        fake_label = fake_label.type_as(fake_pred)

                    loss_cc = c_loss(real_pred, real_label)  # classifier loss
                    loss_cg = c_loss(fake_pred, fake_label)  # downstream loss for generator

                    optimizerG.zero_grad()
                    loss_cg.backward()
                    optimizerG.step()

                    optimizerC.zero_grad()
                    loss_cc.backward()
                    optimizerC.step()

            generator_loss = loss_g.detach().cpu()
            cross_entropy_loss = cross_entropy.detach().cpu()
            discriminator_loss = loss_d.detach().cpu()
            grad_pen = pen.detach().cpu()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i+1],
                'Generator Loss': [generator_loss.item()],
                'Cross Entropy Loss': [cross_entropy_loss.item()],
                'Discriminator Loss': [discriminator_loss.item()],
                'Gradient Penalty': [grad_pen.item()],
            })
            if task:
                downstream_loss = loss_cg.detach().cpu()
                classifier_loss = loss_cc.detach().cpu()
                epoch_loss_df['Downstream Loss'] = [downstream_loss.item()]
                epoch_loss_df['Classifier Loss'] = [classifier_loss.item()]
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )

        if return_loss:
            return self.loss_values

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self._generator.eval()

        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size, sk=self.sk)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            with torch.no_grad():
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def conditional_sample(self, target: str, classes_count: dict, n=1000):
        synthetic_data = pd.DataFrame()

        for c, count in classes_count.items():
            curr_count = 0
            while True:
                samples = self.sample(n, target, c)
                if c in samples[target].unique():
                    class_cond_samples = samples.loc[samples[target] == c]
                    if (curr_count + len(class_cond_samples)) >= count:
                        synthetic_data = pd.concat([synthetic_data, class_cond_samples.head(count-curr_count)], axis=0)
                        break
                    else:  # keep adding until curr_count = count for certain class of target variable
                        synthetic_data = pd.concat([synthetic_data, class_cond_samples], axis=0)
                        curr_count += len(class_cond_samples)

        return synthetic_data

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
