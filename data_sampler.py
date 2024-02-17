"""DataSampler module."""

import numpy as np


class DataSampler(object):
    """DataSampler samples the conditional vector and corresponding data for CTGAN."""

    def __init__(self, data, output_info, log_frequency, transformed_discrete_columns):
        self.transformed_discrete_columns = transformed_discrete_columns
        self._data_length = len(data)

        def is_discrete_column(column_info):
            return (len(column_info) == 1
                    and column_info[0].activation_fn == 'softmax')

        n_discrete_columns = sum(
            [1 for _, column_info in output_info if is_discrete_column(column_info)])

        self._discrete_column_matrix_st = np.zeros(
            n_discrete_columns, dtype='int32')

        # Store the row id for each category in each discrete column.
        # For example _rid_by_cat_cols[a][b] is a list of all rows with the
        # a-th discrete column equal value b.
        self._rid_by_cat_cols = []

        # Compute _rid_by_cat_cols
        st = 0
        for _, column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim

                rid_by_cat = []
                for j in range(span_info.dim):
                    rid_by_cat.append(np.nonzero(data[:, st + j])[0])
                self._rid_by_cat_cols.append(rid_by_cat)
                st = ed
            else:
                st += sum([span_info.dim for span_info in column_info])
        assert st == data.shape[1]

        # Prepare an interval matrix for efficiently sample conditional vector
        max_category = max([
            column_info[0].dim
            for _, column_info in output_info
            if is_discrete_column(column_info)
        ], default=0)

        self._discrete_column_cond_st = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_n_category = np.zeros(n_discrete_columns, dtype='int32')
        self._discrete_column_category_prob = np.zeros((n_discrete_columns, max_category))
        self._n_discrete_columns = n_discrete_columns
        self._n_categories = sum([
            column_info[0].dim
            for _, column_info in output_info
            if is_discrete_column(column_info)
        ])  # equal len(self.transformed_discrete_columns)

        st = 0
        current_id = 0
        current_cond_st = 0
        self.discrete_columns = []
        for column_name, column_info in output_info:
            if is_discrete_column(column_info):
                span_info = column_info[0]
                ed = st + span_info.dim
                category_freq = np.sum(data[:, st:ed], axis=0)
                if log_frequency:
                    category_freq = np.log(category_freq + 1)
                category_prob = category_freq / np.sum(category_freq)
                self._discrete_column_category_prob[current_id, :span_info.dim] = category_prob
                self._discrete_column_matrix_st[current_id] = st
                self._discrete_column_cond_st[current_id] = current_cond_st
                self._discrete_column_n_category[current_id] = span_info.dim
                current_cond_st += span_info.dim
                current_id += 1
                st = ed
                self.discrete_columns.append(column_name)
            else:
                st += sum([span_info.dim for span_info in column_info])

    def _random_choice_prob_index(self, discrete_column_id):
        probs = self._discrete_column_category_prob[discrete_column_id]
        r = np.expand_dims(np.random.rand(probs.shape[0]), axis=1)
        return (probs.cumsum(axis=1) > r).argmax(axis=1)

    def extract_category(self, col_idxs, target_id):
        discrete_column_id = []
        for col_idx in col_idxs:
            is_target = self._discrete_column_cond_st > col_idx
            if np.any(is_target):
                discrete_column_id.append(np.argmax(is_target)-1)
            else:
                discrete_column_id.append(target_id)
        return np.array(discrete_column_id), col_idxs - self._discrete_column_cond_st[discrete_column_id]

    def enforce_skip_constraints(self, batch, cond, mask, discrete_column_id, category_id_in_col):
        """
        NOTE: this is demo for enforcing ONE skip constraint.
        Implementation for enforcing all skip constraints cannot be publicly released yet due to data sharing restrictions.
        """
        sk_mask = np.zeros(self._n_discrete_columns)
        constrained_columns = [f'ID{j}' for j in [1,3,4,5,6,7,8,9,10,11,15,16,17,18,19,20]]
        sk_idx = [idx for idx, c in enumerate(self.discrete_columns) if c in constrained_columns]
        sk_mask[sk_idx] = 1
        for i in range(batch):
            if (discrete_column_id[i] in sk_idx and \
                ((discrete_column_id[i] != self.discrete_columns.index('ID15') and category_id_in_col[i] == 0) or \
                 (discrete_column_id[i] == self.discrete_columns.index('ID15') and category_id_in_col[i] == self._discrete_column_n_category[discrete_column_id[i]]-1))
            ):
                if mask is not None:    mask[i, :] = sk_mask
                cond[i, [self.transformed_discrete_columns.index(f'{c}_0') for c in constrained_columns[:-6]+['ID16']]] = 1
                cond[i, [self.transformed_discrete_columns.index(f'{c}_-1') for c in constrained_columns[-4:]]] = 1
                cond[i, self.transformed_discrete_columns.index('ID15_5')] = 1
            # elif discrete_column_id[i] in []:  # next set of constraints
            #     ...
            else:
                cond[i, self._discrete_column_cond_st[discrete_column_id[i]] + category_id_in_col[i]] = 1  # select specifc mode/class of some chosen variable

        return cond, mask

    def sample_condvec(self, batch, q=1, sk=False, col_idxs=None):
        """Generate the conditional vector for training.

        Returns:
            cond (batch x #categories):
                The conditional vector.
            mask (batch x #discrete columns):
                A one-hot vector indicating the selected discrete column.
            discrete column id (batch):
                Integer representation of mask.
            category_id_in_col (batch):
                Selected category in the selected discrete column.
        """
        if self._n_discrete_columns == 0:
            return None

        cond = np.zeros((batch, self._n_categories), dtype='float32')
        mask = np.zeros((batch, self._n_discrete_columns), dtype='float32')

        target_id = self._n_discrete_columns - 1
        if col_idxs is not None:
            col_idxs = np.append(col_idxs,
                                 self._discrete_column_cond_st[target_id]+self._random_choice_prob_index(target_id*np.ones((batch//q)-col_idxs.size, dtype=int))
            )
            col_idxs = np.repeat(np.sort(col_idxs), q)
            discrete_column_id, category_id_in_col = self.extract_category(col_idxs, target_id)
        else:
            discrete_column_id = np.random.choice(
                np.arange(self._n_discrete_columns), batch)
            category_id_in_col = self._random_choice_prob_index(discrete_column_id)

        mask[np.arange(batch), discrete_column_id] = 1
        if sk:
            cond, mask = self.enforce_skip_constraints(batch, cond, mask, discrete_column_id, category_id_in_col)
        else:
            category_id = (self._discrete_column_cond_st[discrete_column_id] + category_id_in_col)
            cond[np.arange(batch), category_id] = 1

        return cond, mask, discrete_column_id, category_id_in_col, discrete_column_id == target_id

    def sample_original_condvec(self, batch, sk=False):
        """Generate the conditional vector for generation use original frequency."""
        if self._n_discrete_columns == 0:
            return None

        category_freq = self._discrete_column_category_prob.flatten()
        category_freq = category_freq[category_freq != 0]  # has length of self._n_categories
        category_freq = category_freq / np.sum(category_freq)
        col_idxs = np.random.choice(np.arange(len(category_freq)), batch, p=category_freq)
        cond = np.zeros((batch, self._n_categories), dtype='float32')
        if sk:
            discrete_column_id, category_id_in_col = self.extract_category(col_idxs)
            cond, _ = self.enforce_skip_constraints(batch, cond, None, discrete_column_id, category_id_in_col)
        else:
            cond[np.arange(batch), col_idxs] = 1

        return cond

    def sample_data(self, data, n, col, opt):
        """Sample data from original training data satisfying the sampled conditional vector.

        Args:
            data:
                The training data.
        Returns:
            n:
                n rows of matrix data.
        """
        if col is None:
            idx = np.random.randint(len(data), size=n)
            return data[idx]

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self._rid_by_cat_cols[c][o]))

        return data[idx]

    def dim_cond_vec(self):
        """Return the total number of categories."""
        return self._n_categories

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        """Generate the condition vector."""
        vec = np.zeros((batch, self._n_categories), dtype='float32')
        id_ = self._discrete_column_cond_st[condition_info['discrete_column_id']]
        id_ += condition_info['value_id']
        vec[:, id_] = 1
        return vec
