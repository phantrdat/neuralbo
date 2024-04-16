import os
import random
from random import sample
from tkinter.messagebox import NO

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import copy
import math
import time
from torch.func import functional_call, vmap, grad
import numpy as np
import torch
import torch.nn as nn

from utils.base_networks import BaseNN

class NeuralBO():
	"""Thompson Sampling using Deep Neural Networks.
	"""
	def __init__(self, cfg):

		# L2 regularization strength
		self.dim = cfg.dimension
		self.n_iters = cfg.n_iters
		self.activation = cfg.activation
		self.weight_decay = cfg.weight_decay

		self.update_cycle = cfg.update_cycle
		
		# hidden size of the NN layers
		self.hidden_size = cfg.W
		# number of layers
		self.n_layers = cfg.L

		# NN hyper-parameters
		self.learning_rate = cfg.learning_rate
		self.epochs = cfg.epochs

		self.use_cuda = cfg.use_cuda
		self.device = torch.device(0 if torch.cuda.is_available() and self.use_cuda else 'cpu')

		# neural network
		self.model = BaseNN(input_size=self.dim,
						   hidden_size=self.hidden_size,
						   n_layers=self.n_layers,
						   p=0.0,
						   activation=self.activation).to(self.device)
		
		self.init_model = copy.deepcopy(self.model)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
		
		
		self.iteration = 0
		
		self.normalized_inputs = cfg.normalized_inputs
		self.normalized_outputs = cfg.normalized_outputs
		


		# Algorithm objective optimization setting
		self.objective_type = cfg.objective_type
		self.n_init = cfg.n_init
		self.n_raw_samples = cfg.n_raw_samples
  
		# Algorithm specific configs
		self.use_matrix_inversion_appoximation = cfg.use_matrix_inversion_appoximation
		if self.use_matrix_inversion_appoximation:
			self.U = (torch.eye(self.approximator_dim)*self.weight_decay).double().to(self.device)

		else:
			self.U_inv = (torch.eye(self.approximator_dim)/self.weight_decay).double().to(self.device)
		
		self.exploration_coeff = cfg.exploration_coeff
		
		self.X_train = None
		self.Y_train = None
		self.feature_mode = cfg.feature_mode
		# self.run_idx = run_idx
	@property
	def approximator_dim(self):
		"""Sum of the dimensions of all trainable layers in the network.
		"""
		return sum(w.numel() for w in self.model.parameters() if w.requires_grad)

	
	def calculate_gradient(self, x):
		"""Get gradient of network prediction w.r.t network weights.
		"""
		def approx_grad(model):
			params = {k: v.detach() for k, v in model.named_parameters()}
			buffers = {k: v.detach() for k, v in model.named_buffers()}
			def compute_preds(params, buffers, sample):
				preds = functional_call(model, (params, buffers), (sample,))
				return torch.sum(preds, dim=0)
			ft_compute_grad = grad(compute_preds)
			ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0))
			ft_per_sample_grads = ft_compute_sample_grad(params, buffers, x)
		
			grads_approx_fast = []
			all_grads = list(ft_per_sample_grads.values())
			for i in range(len(all_grads)):
				all_grads[i] = all_grads[i].flatten(1)/np.sqrt(self.hidden_size)
			grads_approx_fast = torch.cat(all_grads, dim=1).T
			return grads_approx_fast

		if self.feature_mode=='static':
			grads_approx = approx_grad(self.init_model)
		elif self.feature_mode =='dynamic':
			grads_approx = approx_grad(self.model)
		
		return grads_approx


	def predict(self, X):
		"""Predict reward. """
		# eval mode
		self.model.eval()
		return self.model(X).detach().squeeze()

	def sample_function_value(self, x):
		"""Update confidence bounds and related quantities for all arms.
		"""
		
		grads = self.calculate_gradient(x)

		if self.use_matrix_inversion_appoximation:
			U_inv = 1/torch.diag(self.U,0)
			U_inv = torch.diag(U_inv)
			AinvGT = torch.matmul(grads.T, U_inv)
			GAinvGT = torch.matmul(AinvGT, grads)
			variances = self.weight_decay*torch.diag(GAinvGT)
		else:
			AinvGT = torch.matmul(grads.T, self.U_inv)
			GAinvGT = torch.matmul(AinvGT, grads)
			variances = self.weight_decay*torch.diag(GAinvGT)
	

		posterior_std = self.exploration_coeff*torch.sqrt(variances)
		posterior_mu = self.predict(x)
		
		# estimated combined bound for reward

		r_hat = torch.normal(posterior_mu, posterior_std)
		return r_hat, (posterior_mu, posterior_std)

	
	def train(self, x_train, y_train):
		"""Train neural approximator.
		"""
		# train mode
		x_train = x_train.double()
		y_train = y_train.double()
		self.model.train()
		loss = torch.tensor(0.0)
		for i in (range(self.epochs)):
			y_pred = self.model.forward(x_train).squeeze().double()
			loss = nn.MSELoss()(y_pred, y_train).double()
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
		return loss

		
	def update_A_inv(self, x_t):


		# Implement matrix inversion with different methods
		# 
		
		def inv_sherman_morrison(u, A_inv):
			"""Inverse of a matrix with rank 1 update.
			"""
			Au = torch.matmul(A_inv, u).squeeze(-1)
			A_inv -= torch.outer(Au, Au)/(1+torch.matmul(u.T, Au))
			return A_inv


		grad_approx = self.calculate_gradient(x_t.unsqueeze(0))
		if self.use_matrix_inversion_appoximation ==True:
			self.U = self.U + torch.matmul(grad_approx, grad_approx.T)
		else:
			self.U_inv = inv_sherman_morrison(
				grad_approx,
				self.U_inv
			)
	
	def random_seed(self):
		frac, whole = math.modf(time.time())
		try:
			seed = int(whole/(10000*frac)) + self.iteration
			torch.manual_seed(seed)
		except:
			torch.manual_seed(1111)
     
	def test_mse(self, x_test, target_y_test):
		self.model.eval()
		target_Y_pred = self.model(x_test)
		print(f"- Eval objective model MSE at iteration {self.iteration}:", nn.MSELoss()(target_y_test, target_Y_pred.squeeze()).double().item())
	
 	# def optimize_acquisition(self, X, bounds):
	# 	if self.acqf_optimizer == 'Adam':
	# 		adam_optimizer = torch.optim.Adam([X], lr=0.001)

	# 		for i in range(200):
	# 			adam_optimizer.zero_grad()
	# 			values, _ = self.TS(X)
	# 			objective = values.sum()
	# 			objective.backward()
	# 			adam_optimizer.step()
	# 			for j, (lb, ub) in enumerate(bounds):
	# 				X.data[...,j].clamp_(lb, ub)
					
	# 	if self.acqf_optimizer == 'L-BFGS':
	# 		print("----------------------------------------LBFGS------------------")
	# 		lbfgs_optimizer = torch.optim.LBFGS([X],
	#                 history_size=10, 
	#                 max_iter=4, 
	#                 line_search_fn="strong_wolfe")
	# 		def closure():
	# 			lbfgs_optimizer.zero_grad()
	# 			values, _ = self.TS(X)
	# 			objective = values.sum()
	# 			objective.backward()
	# 			return objective
	# 		for i in range(20):
	# 			lbfgs_optimizer.step(closure)
	# 			for j, (lb, ub) in enumerate(bounds):
	# 				X.data[...,j].clamp_(lb, ub)
				
	# 	samples, (predictive_means, predictive_stds) = self.TS(X)
	# 	return X, samples, (predictive_means, predictive_stds)
	# def optimize_acquisition_discrete(self, X):
	# 	batch_size = 5
		
	# 	chunks = torch.split(X, batch_size)
	# 	acqf_values = []
	# 	for b in chunks:
	# 		acqf_v, (pmean, pvar) =  self.TS(b)
	# 		acqf_values += acqf_v
	# 	acqf_values = torch.stack(acqf_values)
	# 	return X[torch.argmin(acqf_values)], torch.argmin(acqf_values)


	def minimize(self, objective):
		
		torch.manual_seed(0)
		init_points = objective.generate_samples(self.n_init)
		print("Initial points:", init_points)

		self.X_train = init_points['features'].to(self.device)
		self.Y_train = init_points['observations'].to(self.device)

		# Pick the last point of init_points to plot
		if type=='syn':
			optimal_values = [objective.value(self.X_train[-1], is_noise=False).item()]
		else:
			optimal_values = [init_points['observations'][-1].item()]


		objective.max = objective.max.to(self.device)
		objective.min = objective.min.to(self.device)
		X_mean = (objective.max + objective.min)/2
		X_std = torch.abs(objective.max  - X_mean)

		if len(self.X_train) != 0 and len(self.Y_train)!=0:

			print("**Fitting known dataset**")
			if self.normalized_inputs:
				X_train = (self.X_train - X_mean)/X_std
			else:
				X_train = self.X_train
			if self.normalized_outputs:
				print("Training with normalized outputs")
				Y_train = (self.Y_train-self.Y_train.mean())/self.Y_train.std()
			else:
				Y_train = self.Y_train
			loss = self.train(X_train, Y_train)
	
		
		self.model.eval()
		torch.manual_seed(1504)
		x_test = objective.generate_features(10000).to(self.device)	
		y_gt = torch.Tensor([objective.value(x, is_noise=False) for x in x_test]).to(self.device)
		if self.normalized_inputs:
			x_test = (x_test- X_mean)/X_std
		
		self.test_mse(x_test, y_gt)
		
		for T in range(self.n_iters):
			print(f"\n----------NeuralBO - Optimization round {T+1}/{self.n_iters}----------\n")
			
			self.iteration = T+1
			
			X_cand = objective.generate_features(self.n_raw_samples).to(self.device)
			if self.normalized_inputs:
				X_cand = (X_cand - X_mean)/X_std
			
			
			f_hat, (posterior_mu, posterior_std) = self.sample_function_value(X_cand)

			min_idx = torch.argmin(f_hat)
			
			xt_mu, xt_std, xt_acqf = posterior_mu[min_idx], posterior_std[min_idx], f_hat[min_idx]
   
			X_next = X_cand[min_idx]

			self.update_A_inv(X_next)

			if self.normalized_inputs:
				X_next = (X_next*X_std + X_mean).detach()
			else:
				X_next = X_next.detach()


			self.X_train = torch.cat([self.X_train, X_next.clone().unsqueeze(0)])
			

			observation = objective.value(X_next)
			
			if self.objective_type == 'synthetic':
				true_value = objective.value(X_next, is_noise=False)
			else:
				true_value = observation

			self.Y_train = torch.cat([self.Y_train, observation.unsqueeze(0)])

			if (T+1) % self.update_cycle == 0:
				
				print(f"** Train NeuralBO with {self.epochs} epochs")
				
				if self.normalized_inputs:				
					X_train = (self.X_train - X_mean)/X_std
				else:
					X_train = self.X_train
				if self.normalized_outputs:
					Y_train = (self.Y_train-self.Y_train.mean())/self.Y_train.std()
				else:
					Y_train = self.Y_train
				loss = self.train(X_train, Y_train)
				
				self.model.eval()
				
			self.test_mse(x_test, y_gt)
			print(f"\n**** Iteration [{T+1}/{self.n_iters}], current value = {true_value}")
			print(f"\n**** Iteration [{T+1}/{self.n_iters}], xt_prediction = {xt_mu}, xt_std={xt_std}")
			print(f"\n**** Iteration [{T+1}/{self.n_iters}], acqf value = {xt_acqf}")

			optimal_values.append(true_value.item())

			

		return optimal_values
	

	