from matplotlib import pyplot as plt
import matplotlib
import pickle as pkl
import numpy as np
# from objectives import MnistClassifier
import itertools
import random
import os
from tqdm import tqdm
# from neuralucb_gpu import NeuralUCB
import math
import glob
import shutil
from objectives import *
# func_name = "MNISTClassifier"
# func_name = 'Ackley'
# func_name = 'HolderTable'
# func_name = 'Shubert'
# func_name = 'Cosine'




D = 20
RNUM = 1000
func_name = 'Griewank'
REVERSED = False


# D = 2
# RNUM = 200
# func_name = 'Schaffer'
# # fun = Michalewics(dim=D)
# REVERSED = False

# D = 100
# RNUM = 1000
# func_name = 'MNIST_Classifier_Weights_Opt'
# REVERSED = False


# D = 50
# RNUM = 2000
# func_name= 'Ackley'
# REVERSED = False

# D = 60
# RNUM = 2000
# func_name = 'Rover'
# REVERSED = True



directory =  f"results/{func_name}_DIM_{D}_ROUNDS_{RNUM}"


TRIAL = 10
fixed_colors = ['red', 'blue','green', 'black']


# colors = {'RBF': 'navy', 'Matern': 'darkred', 'Linear': 'slategray', 'UCB': 'black'}
# colors = matplotlib.cm.tab20(range(20))
all_algs = ['NeuralBO', 'PINN_NeuralBO', 'NeuralTS-UCB', 'DNGO', 'RF', 'GPEI_RBF', 'GPEI_Matern',  'GPUCB', 'GPUCB_Matern', 'GPTS', 'GPTS_Matern', 'NeuralGreedy']
colors_map = {'NeuralBO': 'red', 'NeuralGreedy': 'blue', 'PINN_NeuralBO': 'brown', 'NeuralBO_PINN_pde_only': 'grey',  'DNGO':'forestgreen', 
				'RF': 'black', 'GPEI': 'grey', 'GPUCB':'purple', 'GPTS': 'gold'}
markers = {'NeuralBO':"*", 'NeuralBO_static':">", 'PINN_NeuralBO': "v", "GPEI": "^", "GPTS":"s",  'GPUCB': "o",
			"RF":"+", "DNGO":"x", 'NeuralGreedy': '.',  'NeuralBO_PINN_pde_only': '<'}
# for i, x in enumerate(all_algs):
	# colors_map[x] = colors[i] 
# colors_map['NeuralTS'] = 'red'
def plot_res(directory):

	norm_z = 1
	
	plotting_objs = {'mean':[],'std':[], 'colors': [], 'labels':[], 'markers':[]}
	# algs =  ["GPEI_Matern", 'NeuralBO', "GPTS_Matern", 'GPUCB_Matern',
	# 		 "RF" , "DNGO", 'NeuralGreedy' ]
	algs = ['PINN_NeuralBO', 'NeuralGreedy']
	# algs = ["NeuralBO", "NeuralGreedy", "GPTS", 'GPUCB', "GPEI"]
	# 'GPTS', 'GPUCB',"GPEI_RBF",
	# algs  = ['GPUCB_Matern', "GPEI_Matern"]
	for alg in algs: 	
		print(alg)
		res_files = glob.glob(f'{directory}/{alg}/*')

		all_runs = np.empty((0, RNUM+1))
		for pkl_file in res_files[:TRIAL]:
			print(pkl_file)
			Di  = pkl.load(open(pkl_file,'rb'))
			
			# if alg=='NeuralBO':
			# 	Di['optimal_values'][0] = Di['optimal_values'][0].cpu().numpy()
			# Di['optimal_values'][0] = -Di['optimal_values'][0]
			# pkl.dump(Di, open(pkl_file, 'wb'))
			optimum_each_run = np.array(Di['optimal_values'])
			idx = np.argmin(Di['optimal_values'])
			# X = Di['X_train'][idx+D]
			# print(fun.value(X, is_noise=False), Di['optimal_values'][idx])
			# optimum_each_run[0] = 2500
			optimum_each_run = np.array([np.min(optimum_each_run[:i]) for i in range(1, RNUM + 2)])			
			all_runs = np.vstack((all_runs, optimum_each_run))
		runs_std = np.std(all_runs, 0)
		runs_mean = np.mean(all_runs, 0)

		if REVERSED ==True:
			runs_mean = np.array([-v for v in runs_mean])

		plotting_objs['mean'].append(runs_mean)
		plotting_objs['std'].append(runs_std)
		plotting_objs['colors'].append(colors_map[alg])
		plotting_objs['markers'].append(markers[alg])
		# if alg.find('_')!=-1:

		# 	plotting_objs['labels'].append(alg[:alg.find('_')])
		# else:
		plotting_objs['labels'].append(alg)

	for (mean, std, color, label, marker) in zip(plotting_objs['mean'], plotting_objs['std'], plotting_objs['colors'], 
										plotting_objs['labels'], plotting_objs['markers']):
		std =  std[:RNUM+1]
		mean = mean[:RNUM+1]
		start = 0
		end = 1001
		plt.plot(np.arange(start,end), mean[start:end], label=label, color=color, marker=marker, markevery=RNUM//25, markersize=5)
		plt.fill_between(np.arange(start,end), (mean - norm_z*std)[start:end], (mean + norm_z*std)[start:end], alpha=0.1, color=color)
	plt.title(f"{func_name} ({D})", fontsize=15)
	fig = plt.gcf()
	size = fig.get_size_inches()*fig.dpi 
	

	leg = plt.legend(fontsize=10, bbox_to_anchor=(0.5, -0.5),loc='lower center', ncol=4)
	for legobj in leg.legendHandles:
		legobj.set_linewidth(3.0)
	
	if func_name == 'NIPS_Text':
		label = 'L1 Distance'
	elif func_name == 'MNIST_Classifier_Weights_Opt':
		label = 'Validation Loss'
	if func_name =='Levy' or func_name =='Ackley' or func_name =='Michalewics' or func_name =='Branin':
		label = "Minimum value observed"
	if func_name == 'Amazon Product Reviews':
		label = "Hierachical F1 score"
	if func_name == 'Robot':
		label = "Reward"
	# if REVERSED == True:
	# 	label = "Evaluation Accuracy"
	# else:
	# 	label = "Minimum value observed"
	plt.ylabel(label, fontsize=15)
	plt.xlabel('Number of evaluations', fontsize=15)
	plt.grid()

	fig.tight_layout() 
	if os.path.isdir("figures") ==False:
		os.makedirs("figures")
	fig.savefig(f'figures/{func_name}_dim_{D}_round_{RNUM}.pdf', dpi=300,  bbox_inches='tight',pad_inches = 0.1)
	plt.savefig(f'figures/{func_name}_dim_{D}_round_{RNUM}.png', dpi=300,  bbox_inches='tight',pad_inches = 0.1)
	print()
	# plt.clf()

def plot_mem_used():

	
	# f=open('logs/Robot/DIM_14/slurm_55739.out', 'r')
	# lines = [l  for l in f.readlines() if l.find("GPU memory used")!=-1]
	# f1 = open('logs/Robot/DIM_14/slurm_55739_num.out','w')
	# f1.writelines(lines)
	# 
	# X = np.array(X)/1024


	f=open('logs/Robot/DIM_14/slurm_55735_num.out', 'r')
	# X = [float(l.strip('\n')) for l in f.readlines()]
	# X = np.array([0]*6001)
	# val = [float(l.strip('\n'))  for l in f.readlines()]
	# index = list(range(0, 6001, 10))
	# X[index] = val
	# for i in range(0, 6001):
	# 	if X[i]==0:
	# 		X[i] = X[i-1]

	# np.save('results/mem_test/NeuralBO_Robot.npy', X)

	Y  = np.arange(0, 5800)
	algs = ["NeuralBO", "NeuralGreedy"]
	colors_map = {'NeuralBO': 'red', 'NeuralGreedy': 'blue', 'GP':'purple'}
	for alg in algs:
		X = np.load(f'results/mem_test/{alg}_Robot.npy')

		plt.plot(Y, X[:5800], label =alg, markevery=100)
	plt.xlabel('Iterations')
	plt.ylabel('Memory used (GB)')
	plt.title("Memory used by baselines for optimizing Robot 14D function")
	plt.legend()
	plt.savefig('test_mem.png', dpi=300,  bbox_inches='tight',pad_inches = 0.1)
	plt.savefig('test_mem.pdf', dpi=300,  bbox_inches='tight',pad_inches = 0.1)

if __name__ == '__main__':
	
	plot_res(directory)
	# plot_mem_used()