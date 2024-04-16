from numpy.random import normal
import torch.nn as nn
import numpy as np
import torch


class BaseNN(nn.Module):
	"""Fully connected neural network for scalar approximation.
	"""
	def __init__(self,
				 input_size=1,
				 hidden_size=2,
				 n_layers=1,
				 activation='ReLU',
				 p=0.0,
				 ):
		super(BaseNN, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.dropout = nn.Dropout(p=p)
		if self.n_layers == 1:
			self.layers = [nn.Linear(input_size, 1, bias=True, dtype=torch.double)]
		else:
			size = [input_size] + [hidden_size, ] * (self.n_layers-1) + [1]
			self.layers = [nn.Linear(size[i], size[i+1],dtype=torch.double) for i in range(self.n_layers)]

		for i in range (0, self.n_layers-1):
			# nn.init.xavier_normal_(self.layers[i].weight)
			nn.init.normal_(self.layers[i].weight,mean=0, std=1/np.sqrt(self.hidden_size))
		
		self.layers = nn.ModuleList(self.layers)
		if activation.lower() == 'sigmoid':
			self.activation = nn.Sigmoid()
		elif activation.lower() == 'relu':
			self.activation = nn.ReLU()
		elif activation.lower() == 'leakyrelu':	
			self.activation = nn.LeakyReLU(negative_slope=0.1)
		elif activation.lower() == 'tanh':
			self.activation = nn.Tanh()

		else:
			raise Exception('{} not an available activation'.format(activation))
	def forward(self, x):
		for i in range(self.n_layers-1):
			# x = self.dropout(self.activation(self.layers[i](x)))
			x = self.activation(self.layers[i](x))
			# x = self.layers[i](x)
		
		x = self.layers[-1](x)
		x = np.sqrt(self.hidden_size)*x
		return x
			# for i in range (0, self.n_layers-1):
			# 	nn.init.kaiming_normal_(self.layers[i].weight, mode='fan_out', nonlinearity='relu')
			# 	# self.layers[i].weight.data.normal_(0,1)
			# self.layers[n_layers-1].weight.data.zero_()
			# for i in range (0, self.n_layers):
				# self.layers[i].weight.data.zero_()
				# if i< self.n_layers-1:
					# for k in range(0, size[i+1]//2):
						# for entry in range(0, size[i]//2):
							# self.layers[i].weight.data[k][entry] = torch.Tensor(np.random.normal(0, 4/hidden_size, 1))
							# self.layers[i].weight.data[size[i+1]//2 +k][size[i]//2+entry] = self.layers[i].weight.data[k][entry]
				# else:
					# for k in range(0, size[i]//2):
					# 	self.layers[i].weight.data[0][k] =  torch.Tensor(np.random.normal(0, 2/hidden_size, 1))
					# 	self.layers[i].weight.data[0][size[i]//2+k] = -self.layers[i].weight.data[0][k]
					
					# for k in range(0, size[i]):
					# 	self.layers[i].weight.data[0][k] =  torch.Tensor(0)
		

		# dropout layer
		# self.dropout = nn.Dropout(p=p)

		# activation function
		

# class BaseNN(nn.Module):
# 	def __init__(self, input_size, hidden_size=2,
# 				 n_layers=1,
# 				 activation='ReLU',
# 				 p=0.8,):
# 		super(BaseNN, self).__init__()
		
# 		self.layer_1 = nn.Linear(input_size, 50)
# 		self.layer_2 = nn.Linear(50, 50, bias=False)
# 		# self.layer_3 = nn.Linear(50, 50, bias=False)
# 		self.layer_out = nn.Linear(50, 1, bias=False)
		
# 		self.relu = nn.ReLU()


# 	def forward(self, inputs):
# 		x = self.relu(self.layer_1(inputs))
# 		x = self.relu(self.layer_2(x))
# 		# x = self.relu(self.layer_3(x))
# 		x = self.layer_out(x)
# 		return (x)
	
# 	def predict(self, test_inputs):
# 		x = self.relu(self.layer_1(test_inputs))
# 		x = self.relu(self.layer_2(x))
# 		x = self.relu(self.layer_3(x))
# 		x = self.layer_out(x)
# 		return (x)

# if __name__=='__main__':
	# model = NeuralNet(input_size=50, hidden_size=200, n_layers=2)
	# epochs = 5000
	# obj = Ackley(50)
	# obj.min = -5.12
	# obj.max = 5.12
	# 

	# device = "cuda:0"
	# dataset = obj.generate_samples(500)
	# X = torch.Tensor(dataset['features']).to(device)
	# Y = torch.Tensor(dataset['observations']).to(device)
	# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=10)
	# train

	# features = X.to(device).float()
	# targets = Y.to(device).float()
	# model.to(device)
	# model.train()

	# for i in tqdm(range(epochs)):
			# optimizer.zero_grad()
			# y_pred = model.forward(features).squeeze().float()
			# loss = nn.MSELoss()(y_pred, targets).float() 
		# 
			# loss.backward()
			# optimizer.step()

	# X_test = torch.Tensor(obj.generate_features(100)).to(device).float()

	# model.eval()
	# output = model.forward(X_test)
	# print([x.item() for x in output])
	# torch.save(model.state_dict(), "test.pth")
	


	
