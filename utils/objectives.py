from abc import abstractmethod
import numpy as np
import torch
import torch
from torch.quasirandom import SobolEngine
import time

class Objective():
	def __init__(self, dim, **kwargs) -> None:
		self.dim = dim
		self.min = torch.full([dim],0)
		self.max = torch.full([dim],1)

		if 'noise_std' in kwargs:
			self.noise_std = kwargs['noise_std']
		else:
			self.noise_std = None
		self.features = None
		self.target_observations = None
	
	def noise_std_estimate(self):
		torch.manual_seed(2609)
		points = self.generate_features(10000)
		v = torch.FloatTensor([self.value(x, is_noise=False) for x in points])
		return 0.1*torch.std(v)
	
	def generate_features(self, sample_num):
		# self.features = torch.FloatTensor(sample_num,self.dim).uniform_(self.min ,self.max)
		# self.features = torch.rand(sample_num,self.dim).to(self.min.device)

		sobol = SobolEngine(dimension=self.dim, scramble=True, seed=int(time.time()%100000))
		self.features = sobol.draw(n=sample_num).to(device=self.min.device)
		self.features = torch.mul(self.max-self.min, self.features) + self.min
		return self.features.double()

	def generate_samples(self, sample_num):
		self.generate_features(sample_num)
		self.target_observations = []
		self.constraint_observations = []
		
		for x in self.features: 
			self.target_observations.append(self.value(x)) 
		self.observations = torch.DoubleTensor(self.target_observations)
		points = {'features': self.features, 'observations':self.observations}

		return points			

	@abstractmethod  
	def value(self, X, is_noise):
		pass


	@property
	def func_name(self):
		return self.__class__.__name__
	
class Ackley(Objective):
	def __init__(self, dim) -> None:
		super().__init__(dim)
		self.min = torch.full([dim],-32.768)
		self.max = torch.full([dim], 32.768)
		if self.noise_std == None:
			self.noise_std = self.noise_std_estimate()
	
	def value(self, X, is_noise=True):

		if is_noise:
			noise = torch.normal(mean=0.0, std=self.noise_std)
		else:
			noise = 0
		
		dim = X.shape[0]
		delta = 1/dim
		a = 20
		b = 0.2 
		c = 2*np.pi
		target =  -a * torch.exp(-b * torch.sqrt(delta * torch.sum(X**2))) - torch.exp(delta * torch.sum(torch.cos(c * X))) + torch.e + a + noise
		return target
	
