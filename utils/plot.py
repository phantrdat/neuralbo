import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
import imageio
import os
import shutil
import pickle as pkl
def Gardner(X,Y):
	return [np.sin(X) * np.sin(Y)]

def Gramacy(X,Y):
	c0 = - 0.5*np.sin(2*np.pi*(X**2 - 2*Y)) - X - 2*Y
	c1 = X**2 + Y**2
	return  [c0, c1]

def Branin(X,Y):
	return [(X-2.5)**2 + (Y-7.5)**2] 

func = {"gardner": Gardner, 'gramacy': Gramacy, 'branin': Branin}
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




def plot_2d_constrained_optimization(space, optimal_pkl_file, constraint_boundaries, func_name):
	min_x, max_x, min_y, max_y = space
	# Generate sample data
	x = np.linspace(min_x, max_x, 100)  # 2D array
	y = np.linspace(min_y, max_y, 100)     # 1D array
	X, Y = np.meshgrid(x, y)      # Creating grid from 1D arrays

	   # Sample 2D data for the colormap
	# infeasible_pts_idx = [np.where(Z > cb)[0] for cb in constraint_boundaries] 
	# infeasible_idx = list(set.intersection(*map(set,infeasible_pts_idx)))
	# infeasible_idx = sorted([f for f in infeasible_idx])
	# Z[infeasible_pts_idx] = 0

	# Create colormap plot
	plt.figure(figsize=(8, 6))
	
	constraints = func[func_name](X,Y)
	feasible_masks =  [constraints[i] - constraint_boundaries[i] <=0  for i in range(len(constraint_boundaries))]
	
	overall_feasible_mask = np.all(feasible_masks, axis=0)
	overall_infeasible_mask = ~overall_feasible_mask
	# for i, R in enumerate(feasible_regions):
		
	plt.contourf(X,Y, overall_feasible_mask, levels=[-0.5, 0.5], colors='white', alpha=0.5)
	plt.contourf(X, Y, overall_infeasible_mask, colors='green',  levels=[-0.5, 0.5])
	
	for i, constraint in enumerate(constraints):
		plt.contour(X, Y, constraint - constraint_boundaries[i], levels=[0], colors='blue', linestyles='dashed')

	plt.xlabel('X1')
	plt.ylabel('x2')
	plt.xlim(left=min_x)
	plt.ylim(bottom=min_y)
	plt.title(f'2D Constrained Optimization - {func_name}')
	plt.tight_layout()
	plt.savefig(f'feasible_regions.png')
	D = pkl.load(
			open(optimal_pkl_file,'rb'), encoding='utf-8')

	optimal_points = D['X_train'][-100:].numpy()
	optimal_values = D['optimal_values'][-100:]

	is_feasible =  [(torch.FloatTensor(v)<=0).all() for v in D['constraint_values']]
	feasible_pts = optimal_points[is_feasible]
	# 
	feasible_values = np.array(optimal_values)[is_feasible]
	minimal_value = None
	if feasible_values.shape[0] !=0:
		# print("feasible found:", feasible_pts)

		min_idx = np.argmin(feasible_values)
		minimal_value = feasible_values[min_idx]
		minimum = feasible_pts[min_idx]
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
				color = "purple"
				marker_size = 5
			plt.scatter(x1, x2, s=marker_size, color=color, marker=marker)
			plt.plot()
			plt.savefig(f'figures/{k:03d}.png', dpi=300)
			if k== len(X1)-1:
				plt.savefig(f'final_figures/{func_name}_{gif_id}.png', dpi=300)
	# 
	# save last figures 
	
	gif_id = optimal_pkl_file.split('.')[1]
	create_gif("figures/",f'{func_name}_c_{gif_id}.gif', duration=300)
	shutil.rmtree('figures')
	
	plt.clf()


if __name__ == '__main__':
	# space = (-0.5, 6.5, -0.5, 6.5)
	# constraint_boundaries = [-0.95]
	# func_name = "gardner"

	# space = (-0.1, 1.1, -0.1, 1.1)
	# constraint_boundaries = [-1.5, 1.5]
	# func_name = "gramacy"

	space = (-5, 10, 0, 15)
	constraint_boundaries = [50]
	func_name = "branin"




	for i in range(8,9):
		# optimum_pkl = f'results/Gramacy_DIM_2_ROUNDS_100/ConstrainedNeuralBO/ConstrainedNeuralBO_L2_w100_Gramacy_dim2_dynamic_matapproxFalse.{i}.pkl'
		optimum_pkl = f"results/BraninHoo_DIM_2_ROUNDS_100/ConstrainedNeuralBO/ConstrainedNeuralBO_BraninHoo_dim2.{i}.pkl"
		plot_2d_constrained_optimization(space, optimum_pkl, constraint_boundaries, func_name)

