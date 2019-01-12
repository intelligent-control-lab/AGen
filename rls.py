import numpy as np
import scipy.linalg
import pdb
import matplotlib.pyplot as plt

class rls(object):
	"""docstring for ClassName"""
	def __init__(self, lbd, theta, nn_dim, output_dim):

		self.lbd = lbd
		self.nn_dim = nn_dim
		self.y_dim = output_dim
		self.draw = []

		self.theta = theta*np.ones([self.nn_dim, self.y_dim])

		self.initialize()


	def initialize(self):
		self.rls_state = []

		self.F = 10000*np.eye(self.nn_dim)

		self.F_M = self.F

		for i in range(self.y_dim - 1):
			self.F_M = scipy.linalg.block_diag(self.F_M, self.F)


	# rls.update(hidden_vec, obs_Y[i,:])
	def update(self, hidden_vec, obs_Y):
		hidden_vec = np.concatenate([hidden_vec, np.ones([1,1])], axis=1)

		for j in range(self.y_dim):
			self.F = self.F_M[self.nn_dim*j:self.nn_dim*(j+1), 
							self.nn_dim*j:self.nn_dim*(j+1)]
			
			k = self.lbd + hidden_vec @ self.F @ hidden_vec.T
			# 65 * 1
			k = self.F @ hidden_vec.T / k

			self.theta[:,j] = self.theta[:,j] + k@(obs_Y[:,j]- hidden_vec@self.theta[:,j])

			self.F_M[self.nn_dim*j:self.nn_dim*(j+1), 
							self.nn_dim*j:self.nn_dim*(j+1)] = self.F

		pred = hidden_vec @ self.theta
		
		error = obs_Y - pred 
		
		self.rls_state.append(pred)


	def predict(self, hidden_vec):
		hidden_vec = np.concatenate([hidden_vec, np.ones([1,1])], axis=1)

		return hidden_vec @ self.theta



if __name__ == '__main__':
	theta = 0
	model = rls(0.99, theta)
	features = np.ones([1,64])
	observeY = np.ones([9500, 2])
	for i in range(observeY.shape[0]):
		model.update(features, observeY[i,:])