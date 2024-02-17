import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class WeightPredictorNetwork(nn.Module):
	def __init__(self, wpn_layers, embedding_matrix):
		"""
		WPN outputs a "virtual" weight matrix W

		:param nn.Tensor(D, M) embedding_matrix: matrix with the embeddings (D = number of features, M = embedding size)
		"""
		super().__init__()
		self.register_buffer('embedding_matrix', embedding_matrix)  # store the static embedding_matrix

		layers = []
		prev_dimension = embedding_matrix.size(1)
		for i, dim in enumerate(wpn_layers):
			if i == len(wpn_layers)-1: # last layer
				layer = nn.Linear(prev_dimension, dim)
				nn.init.uniform_(layer.weight, -0.01, 0.01) 	# same initialization as in the Diet Network paper official implementation
				layers.append(layer)
				layers.append(nn.Tanh())
			else:
				layer = nn.Linear(prev_dimension, dim)
				nn.init.kaiming_normal_(layer.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
				layers.append(layer)
				layers.append(nn.LeakyReLU())
				layers.append(nn.BatchNorm1d(dim))
				layers.append(nn.Dropout(0.2))

			prev_dimension = dim

		self.wpn = nn.Sequential(*layers)

	def forward(self):
		W = self.wpn(self.embedding_matrix) 	# W has size (D x K)

		return W.T # size K x D


class SparsityNetwork(nn.Module):
	"""
	Sparsity network
	- same architecture as WPN
	- input: gene embedding matrix (D x M)
	- output: 1 neuron, sigmoid activation function (which will get multiplied by the weights associated with the gene)
	"""

	def __init__(self, wpn_layers, embedding_matrix):
		"""
		:param nn.Tensor(D, M) embedding_matrix: matrix with the embeddings (D = number of features, M = embedding size)
		"""
		super().__init__()
		self.register_buffer('embedding_matrix', embedding_matrix)  # store the static embedding_matrix

		layers = []
		dim_prev = embedding_matrix.size(1)  # input for global sparsity: gene embedding
		for _, dim in enumerate(wpn_layers):
			layers.append(nn.Linear(dim_prev, dim))
			layers.append(nn.LeakyReLU())
			layers.append(nn.BatchNorm1d(dim))
			layers.append(nn.Dropout(0.2))

			dim_prev = dim

		layers.append(nn.Linear(dim, 1))
		self.network = nn.Sequential(*layers)

	def forward(self):
		"""
		Input:
		- input: None

		Returns:
		- Tensor of sigmoid values (D)
		"""
		out = self.network(self.embedding_matrix) # (D, 1)
		out = torch.sigmoid(out)
		return torch.squeeze(out, dim=1) 		  # (D)


class FirstLinearLayer(nn.Module):
	"""
	First linear layer (with activation, batchnorm and dropout), with the ability to include:
	- diet layer (i.e., there's a weight predictor network which predicts the weight matrix)
	- sparsity network (i.e., there's a sparsity network which outputs sparsity weights)
	"""
	def __init__(self, wpn_layers, classifier_dim, embedding_matrix):
		"""
		If is_diet_layer==None and sparsity==None, this layers acts as a standard linear layer
		"""
		super().__init__()

		self.wpn = WeightPredictorNetwork(wpn_layers, embedding_matrix)

		# auxiliary layer after the matrix multiplication
		self.bias_first_layer = nn.Parameter(torch.zeros(classifier_dim[0]))
		self.layers_after_matrix_multiplication = nn.Sequential(*[
			nn.LeakyReLU(),
			nn.BatchNorm1d(classifier_dim[0]),
			nn.Dropout(0.2)
		])

		self.sparsity_model = SparsityNetwork(wpn_layers, embedding_matrix)

	def forward(self, x):
		"""
		Input:
			x: (batch_size x num_features)
		"""
		# first layer
		W = self.wpn()  # W has size (K x D)

		all_sparsity_weights = self.sparsity_model()     # Tensor (D, )
		W = torch.matmul(W, torch.diag(all_sparsity_weights))

		hidden_rep = F.linear(x, W, self.bias_first_layer)

		return self.layers_after_matrix_multiplication(hidden_rep), all_sparsity_weights
