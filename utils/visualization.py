import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import imageio
import os
import shutil
import pickle as pkl

def create_gif(image_folder, output_file, duration=0.5):
		"""
		Create a GIF from a folder containing images.
		
		Parameters:
				image_folder (str): Path to the folder containing images.
				output_file (str): Output file name (with .gif extension).
				duration (float): Time duration (in seconds) between frames.
		"""
		images = []
		
		# Iterate through all images in the folder
		for filename in sorted(os.listdir(image_folder)):
				if filename.endswith('.png') or filename.endswith('.jpg'):
						image_path = os.path.join(image_folder, filename)
						images.append(Image.open(image_path))
		
		# Save the images as a GIF
		imageio.mimsave(output_file, images, duration=duration)




def plot_2d_constrained_optimization(space, optimal_pkl_file, func_name):
	min_x, max_x, min_y, max_y = space
	# Generate sample data
	# x = np.linspace(min_x, max_x, 100)  # 2D array
	# y = np.linspace(min_y, max_y, 100)     # 1D array
	# X, Y = np.meshgrid(x, y)      # Creating grid from 1D arrays


	# Create colormap plot
	plt.figure(figsize=(8, 6))
	
	

	plt.xlabel('X1')
	plt.ylabel('x2')
	plt.xlim(left=min_x, right=max_x)
 
	plt.ylim(bottom=min_y, top=max_y)
	plt.title(f'2D Constrained Optimization - {func_name}')
	plt.tight_layout()
	D = pkl.load(
			open(optimal_pkl_file,'rb'), encoding='utf-8')

	optimal_points = D['X_train'].numpy()
	optimal_values = D['optimal_values']



	min_idx = np.argmin(optimal_values)
	minimal_value = optimal_values[min_idx]
	minimum = optimal_points[min_idx]
	print(f"feasible minimum {minimum}, min feasible value {minimal_value}")

	X1 = optimal_points[:,0]
	X2 = optimal_points[:,1]

	gif_id = optimal_pkl_file.split('.')[1]

	if os.path.isdir('figures')==False:
		os.mkdir('figures')
		
	for k, (x1, x2) in enumerate(zip(X1, X2)):
			if optimal_values[k] == minimal_value:
				marker = "*"
				color = "red"
				marker_size = 30
			else: 
				marker = '.'
				color = "blue"
				marker_size = 30
			plt.scatter(x1, x2, s=marker_size, color=color, marker=marker)
			plt.plot()
			file_name = f'figures/{k:03d}.png'
			plt.savefig(file_name, dpi=300)
			if k== len(X1)-1:
				plt.savefig(f'final_figures/{func_name}_{gif_id}.png', dpi=300)
	# 
	# save last figures 
	
	gif_id = optimal_pkl_file.split('.')[1]
	create_gif("figures/",f'{func_name}_c_{gif_id}.gif', duration=300)
	shutil.rmtree('figures')
	
	plt.clf()


if __name__ == '__main__':

	space = (-32.768, 32.768, -32.768, 32.768)
	func_name = "ackley"




	for i in range(1,2):
		optimum_pkl = f"results/Ackley_DIM_2_ITERS_200/NeuralBO/NeuralBO_Ackley_dim2.{i}.pkl"
		plot_2d_constrained_optimization(space, optimum_pkl, func_name)

