import numpy as np
import scipy.linalg
import pdb

class rls(object):
	"""docstring for ClassName"""
	def __init__(self, lbd, theta):

		self.lbd = lbd
		self.nn_dim = 65 # 64 # here you must add bias 
		self.y_dim = 2

		if theta == 0:
			self.theta = np.zeros([self.nn_dim, self.y_dim])
		else:
			# self.theta = theta
			self.theta = theta*np.ones([self.nn_dim, self.y_dim])
			print(self.theta.shape)

		self.initialize()


	def initialize(self):
		# can I get theta from the npz
		# I think so
		self.rls_state = []

		self.F = 10000*np.eye(self.nn_dim)

		self.F_M = self.F

		for i in range(self.y_dim - 1):
			self.F_M = scipy.linalg.block_diag(self.F_M, self.F)


	# rls.update(hidden_vec, obs_Y[i,:])
	def update(self, hidden_vec, obs_Y):
		hidden_vec = np.concatenate([hidden_vec, np.ones([1,1])], axis=1)

		for j in range(self.y_dim):
			#pdb.set_trace()

			self.F = self.F_M[self.nn_dim*j:self.nn_dim*(j+1), 
							self.nn_dim*j:self.nn_dim*(j+1)]
			# 65*65
			#pdb.set_trace()
			# 1*1
			k = self.lbd + hidden_vec @ self.F @ hidden_vec.T
			# 65 * 1
			k = self.F @ hidden_vec.T / k
			print(self.theta.shape)
			print(obs_Y.shape)


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