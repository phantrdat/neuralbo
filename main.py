import pickle as pkl
import os
import argparse
import json
import numpy as np
import time
from baselines.neuralbo import NeuralBO
from types import SimpleNamespace
import warnings
import importlib
def fxn():
	warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
	warnings.simplefilter("ignore")
	fxn()

DATA_DIR = 'data/'
RES_DIR = 'results'
default_json= 'config/Ackley/DIM_2/NeuralBO_ackley2.json'

parser = argparse.ArgumentParser()
parser.add_argument('-cfg', type=str, default=default_json, help='Config File')
parser.add_argument('-gpu_id', default=0, help='GPU ID')	
args = parser.parse_args()


GPU_ID = int(args.gpu_id)
objective_functions = importlib.import_module('utils.objectives')




def run_opt(alg, objective):
	t1 = time.time()
	optimal_values = alg.minimize(objective)
	t2 = time.time() - t1
	info = {'function_name': objective.func_name,
			'function_properties': objective.__dict__, 
			"optimal_values": optimal_values, 
			"dim": objective.dim, 
			"X_train": alg.X_train.cpu(),
			"Y_train": alg.Y_train.cpu(),
			"alg_configs": configs,

			"Running time": t2}
	return info


if __name__ == '__main__':


	configs = json.load(open(args.cfg, "r"))
	print(configs)

	configs = SimpleNamespace(**configs)
		
	
	objective = None
	if configs.objective_type =='synthetic':
		objective = getattr(objective_functions, configs.function_name)(dim=configs.dimension)
	if 'neuralbo' == configs.algorithm_type.lower():		
		print("Normalized outputs:", configs.normalized_outputs)
		print("Normalized inputs:", configs.normalized_inputs)
		print("Use matrix inversion approximation:", configs.use_matrix_inversion_appoximation)
		for run_idx in range(configs.first_run, configs.last_run):
			print("Run:", run_idx)
			
			neuralbo = NeuralBO(cfg=configs)
			info = run_opt(neuralbo, objective)

			save_root = f"results/{info['function_name']}_DIM_{configs.dimension}_ITERS_{configs.n_iters}/{configs.algorithm_type}"
			if os.path.isdir(save_root) ==False:
				os.makedirs(save_root)

			file_name = f"{save_root}/{configs.algorithm_type}_{info['function_name']}_dim{info['dim']}.{f'{run_idx:02d}'}.pkl"
			pkl.dump(info, open(file_name,'wb'))

	
	
	


	




		

	
