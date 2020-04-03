"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

# ZIFA (downloaded in 2019.11.5): https://github.com/epierson9/ZIFA


import os
import argparse
import scanpy.api as sc

from ZIFA import ZIFA
from ZIFA import block_ZIFA

from script.utils_ import get_all_small_data_names, get_all_middle_data_names, get_all_large_data_names
from script.utils_ import get_data, to_adata, from_adata, combine_clt_json_to_csv, run_kmeans_clt
from script.constant import RESULT_PATH, TEMP_PATH


def preprocess(X, y, cell_norm):
	adata = to_adata(X, y)
	if cell_norm:
		sc.pp.normalize_per_cell(adata)
	sc.pp.log1p(adata)
	X, y = from_adata(adata)
	return X.toarray(), y


def run_single_data(data_name, repeat=5, clt_repeat=10, hidden_dim=256, cell_norm=0):
	X, y_true = get_data(data_name)
	X, y_true = preprocess(X, y_true, cell_norm)

	result_folder = os.path.join(RESULT_PATH, f'ZIFA-{hidden_dim}-cell_norm{cell_norm}', data_name)
	os.makedirs(result_folder, exist_ok=True)

	# for repeat_id in range(repeat):
	for repeat_id in [3, 4]:
		print(f'======================================\n{data_name}: repeat = {repeat_id}')
		h, model_params = block_ZIFA.fitModel(X, hidden_dim)
		assert h.shape == (X.shape[0], hidden_dim)
		run_kmeans_clt(h, y_true, os.path.join(result_folder, f'repeat-{repeat_id}.json'), clt_repeat=clt_repeat)
	combine_clt_json_to_csv(inpath=result_folder, outpath=os.path.join(result_folder, 'final.csv'))


def main(args):
	repeat = 5
	clt_repeat = 10

	for data_name in ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell']:
		run_single_data(data_name, repeat, clt_repeat, hidden_dim=args.hidden_dim, cell_norm=args.cell_norm)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--hidden_dim', type=int, default=256)
	parser.add_argument('--cell_norm', type=int, default=0)
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	# c03 cell_norm 0 hidden 64 small data
	# d02 baseline4 cell_norm 0 hidden 2 small data
	main(args)





