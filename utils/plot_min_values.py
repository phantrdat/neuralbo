from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np
import os
import glob
from objectives import *



TRIAL = 10
fixed_colors = ['red']

all_algs = ['NeuralBO']
colors_map = {'NeuralBO': 'red'}
markers = {'NeuralBO':"*"}

def plot_min_values(directory, output_dir='min_values_plot/'):

	norm_z = 1
	plotting_objs = {'mean':[],'std':[], 'colors': [], 'labels':[], 'markers':[]}
	algs = ['NeuralBO']
	
	for alg in algs: 	
		print(alg)
		res_files = sorted(glob.glob(f'{directory}/{alg}/*'))

		all_runs = np.empty((0, N_ITERS+1))
		for pkl_file in res_files[:TRIAL]:
			print(pkl_file)
			Di  = pkl.load(open(pkl_file,'rb'))
			optimum_each_run = np.array(Di['optimal_values'])
			idx = np.argmin(Di['optimal_values'])
			optimum_each_run = np.array([np.min(optimum_each_run[:i]) for i in range(1, N_ITERS + 2)])			
			all_runs = np.vstack((all_runs, optimum_each_run))
		runs_std = np.std(all_runs, 0)
		runs_mean = np.mean(all_runs, 0)

		if REVERSED ==True:
			runs_mean = np.array([-v for v in runs_mean])

		plotting_objs['mean'].append(runs_mean)
		plotting_objs['std'].append(runs_std)
		plotting_objs['colors'].append(colors_map[alg])
		plotting_objs['markers'].append(markers[alg])
		plotting_objs['labels'].append(alg)

	plt.figure(figsize=(8, 6))
	for (mean, std, color, label, marker) in zip(plotting_objs['mean'], plotting_objs['std'], plotting_objs['colors'], 
										plotting_objs['labels'], plotting_objs['markers']):
		std =  std[:N_ITERS+1]
		mean = mean[:N_ITERS+1]
		plt.plot(np.arange(0,N_ITERS+1), mean[0:N_ITERS+1], label=label, color=color, marker=marker, markevery=N_ITERS//25, markersize=5)
		plt.fill_between(np.arange(0,N_ITERS+1), (mean - norm_z*std)[0:N_ITERS+1], (mean + norm_z*std)[0:N_ITERS+1], alpha=0.1, color=color)
	plt.title(f"{func_name} ({D})", fontsize=15)

	

	leg = plt.legend(fontsize=10, bbox_to_anchor=(0.5, -0.5),loc='lower center', ncol=4)
	for legobj in leg.legend_handles:
		legobj.set_linewidth(3.0)
	
	label = "Minimum value observed"
	plt.ylabel(label, fontsize=15)
	plt.xlabel('Number of evaluations', fontsize=15)
	plt.grid()

	plt.tight_layout() 
	if os.path.isdir(output_dir)==False:
		os.mkdir(output_dir)
	plt.savefig(f'{output_dir}/{func_name}_DIM_{D}_NITERS_{N_ITERS}.pdf', dpi=300,  bbox_inches='tight',pad_inches = 0.1)
	plt.savefig(f'{output_dir}/{func_name}_DIM_{D}_NITERS_{N_ITERS}.png', dpi=300,  bbox_inches='tight',pad_inches = 0.1)
	print()

if __name__ == '__main__':
	D = 2
	N_ITERS = 100
	func_name= 'Ackley'
	REVERSED = False
	directory =  f"results/{func_name}_DIM_{D}_ITERS_{N_ITERS}"
	
	plot_min_values(directory)
