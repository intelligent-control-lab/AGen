import numpy as np
import scipy.linalg
import pdb
import matplotlib.pyplot as plt

class rls(object):
	"""docstring for ClassName"""
	def __init__(self, lbd, theta):

		self.lbd = lbd
		self.nn_dim = 131 # 64 # here you must add bias # hidden 64 + observe 66 + bias 1= 131
		self.y_dim = 2
		self.draw = []

		# if theta.all() == 0:
		# 	self.theta = np.zeros([self.nn_dim, self.y_dim])
		# else:
		# 	# self.theta = theta
		self.theta = theta*np.ones([self.nn_dim, self.y_dim])
		print('I\'m here')

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
		# obs_Y should be k+1 time
		# hidden_vec should be k time
		hidden_vec = np.concatenate([hidden_vec, np.ones([1,1])], axis=1)

		for j in range(self.y_dim):
			#pdb.set_trace()

			self.F = self.F_M[self.nn_dim*j:self.nn_dim*(j+1), 
							self.nn_dim*j:self.nn_dim*(j+1)]
			# 65*65
			
			# 1*1
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