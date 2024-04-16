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




def plot_2d_constrained_optimization(optimal_pkl_file, output_dir='visualization/'):
	temp_dir = 'temp/'
	
	if os.path.isdir(temp_dir)==False:
		os.mkdir(temp_dir)
	if os.path.isdir(output_dir)==False:
		os.mkdir(output_dir)

	D = pkl.load(
			open(optimal_pkl_file,'rb'), encoding='utf-8') 
	min_x, min_y = D['function_properties']['min'].cpu().numpy()
	max_x, max_y = D['function_properties']['max'].cpu().numpy()
	func_name = D['function_name']
	
	# Create colormap plot
	plt.figure(figsize=(8, 6))
	
	

	plt.xlabel('X1')
	plt.ylabel('x2')
	plt.xlim(left=min_x, right=max_x)
 
	plt.ylim(bottom=min_y, top=max_y)
	plt.title(f'2D Constrained Optimization - {func_name}')
	plt.tight_layout()
	
	# Get selected points w.r.t optimization steps, ignore initialization points
	n_points = D['alg_configs'].n_iters
	optimal_points = D['X_train'][-n_points:].numpy()
	optimal_values = D['optimal_values'][-n_points:]


	min_idx = np.argmin(optimal_values)
	minimal_value = optimal_values[min_idx]
	minimum = optimal_points[min_idx]
	print(f"feasible minimum {minimum}, min feasible value {minimal_value}")

	X1 = optimal_points[:,0]
	X2 = optimal_points[:,1]

	gif_id = optimal_pkl_file.split('.')[1]

	
		
	for k, (x1, x2) in enumerate(zip(X1, X2)):
		if optimal_values[k] == minimal_value:
			marker = "*"
			color = "red"
			marker_size = 30
			x1_star, x2_star = (x1,x2)
		else: 
			marker = '.'
			color = "blue"
			marker_size = 30
		plt.scatter(x1, x2, s=marker_size, color=color, marker=marker)
		plt.plot()
		file_name = f'temp/{k:03d}.png'
		plt.savefig(file_name, dpi=300)
		if k == len(X1)-1:
			# save last figures
			
   
			plt.scatter(x1_star, x2_star, s=marker_size, color='red', marker='*')
			figure_path = os.path.join(output_dir, f"{func_name}_{gif_id}.png")
			plt.savefig(figure_path, dpi=300)
	
	
	gif_id = optimal_pkl_file.split('.')[1]
	create_gif(temp_dir,os.path.join(output_dir, f'{func_name}_{gif_id}.gif'), duration=300)
	shutil.rmtree('temp')
	
	plt.clf()


if __name__ == '__main__':

	optimum_pkl = f"results/Ackley_DIM_2_ITERS_100/NeuralBO/NeuralBO_Ackley_dim2.01.pkl"
	plot_2d_constrained_optimization(optimum_pkl)

