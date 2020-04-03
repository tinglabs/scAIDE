"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

# SIMLR (downloaded in 2019.11.4): https://github.com/bowang87/SIMLR_PY

import os, sys
import json
import argparse
import scanpy.api as sc
import numpy as np
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi

from script.utils_ import get_all_small_data_names, get_all_middle_data_names, get_all_large_data_names, get_all_sim_sparsity_data_names
from script.utils_ import get_data, to_adata, from_adata, combine_clt_json_to_csv
from script.constant import RESULT_PATH, TEMP_PATH, BASELINE_SRC_PATH

sys.path.append(BASELINE_SRC_PATH + os.sep + 'SIMLR_PY')
import SIMLR


def preprocess(X, y, max_dim, cell_norm, log1p):
	adata = to_adata(X, y)
	if cell_norm:
		sc.pp.normalize_per_cell(adata)
	if log1p:
		sc.pp.log1p(adata)
	X, y = from_adata(adata)
	if X.shape[1] > max_dim:
		X = SIMLR.helper.fast_pca(X, 500)
	else:
		X = X.todense()
	return X, y


def get_input_data(data_name, max_dim, cell_norm, log1p):
	save_folder = os.path.join(TEMP_PATH, 'SIMLR', data_name); os.makedirs(save_folder, exist_ok=True)
	x_npy = os.path.join(save_folder, f'X-{max_dim}-{int(cell_norm)}-{int(log1p)}.npy')
	y_npy = os.path.join(save_folder, f'y_true-{max_dim}-{int(cell_norm)}-{int(log1p)}.npy')
	if os.path.exists(x_npy):
		X = np.load(x_npy)
		y_true = np.load(y_npy) if os.path.exists(y_npy) else None
		return X, y_true
	X, y_true = get_data(data_name)
	X, y_true = preprocess(X, y_true, max_dim=max_dim, cell_norm=cell_norm, log1p=log1p)
	np.save(x_npy, X)
	if y_true is not None:
		np.save(y_npy, y_true)
	return X, y_true


def run_single_data(data_name, repeat=5, clt_repeat=10, max_dim=500, neighbor=30, max_iter=5, cell_norm=0, log1p=1):
	X, y_true = get_input_data(data_name, max_dim, cell_norm, log1p)
	c = len(np.unique(y_true))
	result_folder = os.path.join(
		RESULT_PATH, f'SIMLR-{max_dim}-{neighbor}-{max_iter}-cell_norm{int(cell_norm)}-log1p{int(log1p)}', data_name)
	print(result_folder)
	os.makedirs(result_folder, exist_ok=True)

	for repeat_id in range(repeat):
		print(f'======================================\n{data_name}: repeat = {repeat_id}')
		simlr = SIMLR.SIMLR_LARGE(c, num_of_neighbor=neighbor, mode_of_memory=0, max_iter=max_iter)
		S, F, val, ind = simlr.fit(X)
		print('F', type(F), F.shape)

		ret_list = []
		for i in range(clt_repeat):
			y_pred = simlr.fast_minibatch_kmeans(F, c)
			metric_dict = {'ARI': ari(y_true, y_pred), 'NMI': nmi(y_true, y_pred)}
			ret_list.append(metric_dict)
			print('{}: {}'.format(i, metric_dict))
		json.dump({'sc_scope':ret_list}, open(os.path.join(result_folder, f'repeat-{repeat_id}.json'), 'w'), indent=2)
	combine_clt_json_to_csv(inpath=result_folder, outpath=os.path.join(result_folder, 'final.csv'))


def main(args):
	data_names = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']
	repeat = 5
	clt_repeat = 10

	for data_name in data_names:
		run_single_data(data_name, repeat, clt_repeat, max_dim=args.max_dim, neighbor=args.neighbor,
			max_iter=args.max_iter, cell_norm=args.cell_norm, log1p=args.log1p)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--max_dim', type=int, default=500) # recommended by tutorial
	parser.add_argument('--neighbor', type=int, default=30)
	parser.add_argument('--max_iter', type=int, default=5)
	parser.add_argument('--cell_norm', type=int, default=1)
	parser.add_argument('--log1p', type=int, default=1)
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	# d02 baseline2 cell_norm0/1 large
	main(args)





