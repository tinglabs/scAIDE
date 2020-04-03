"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

# SAUCIE (downloaded in 2019.11.5): https://github.com/KrishnaswamyLab/SAUCIE/
# Note: the source code of scScope is modified for easier running and bug fixing (marked with 'HY modified')
# Poor performance; Drop

import os
import numpy as np
import json
import argparse
import scanpy.api as sc
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
import math

from SAUCIE.model import SAUCIE
from SAUCIE.loader import Loader

from utils_ import get_data, run_kmeans_clt
from utils_ import combine_clt_json_to_csv
from constant import TEMP_PATH, RESULT_PATH
from utils_ import get_all_small_data_names, get_all_middle_data_names, get_all_large_data_names
from utils_ import count_non_zero, to_adata, from_adata, timer


def asinh(x, scale=5.):
	"""Asinh transform."""
	f = np.vectorize(lambda y: math.asinh(y / scale))
	return f(x)


def sinh(x, scale=5.):
	"""Reverse transform for asinh."""
	return scale * np.sinh(x)


@timer
def preprocess(data_name, cell_norm=1):
	X, y_true = get_data(data_name)
	adata = to_adata(X, y_true)
	if cell_norm:
		sc.pp.normalize_per_cell(adata)
	# sc.pp.log1p(adata)
	X, y_true = from_adata(adata)
	X.data = asinh(X.data)  # used in SAUCIE get_data function
	return X.toarray(), y_true


def get_input_data(data_name, imputed=0, cell_norm=1):
	if imputed:
		pass
	else:
		return preprocess(data_name, cell_norm=cell_norm)


def get_imputed_folder(data_name):
	pass


def run_imputed(data_name, lambda_b=0.1):
	"""Note: every .csv stores cell-gene matrix from the same batch"""
	pass


def run_single_data(data_name, repeat_id, imputed=0, cell_norm=1, lambda_c=0.1, lambda_d=0.2, iter_num=4000, layers=[512,256,128,2]):
	X, y_true = get_input_data(data_name, imputed=imputed, cell_norm=cell_norm)
	result_folder = os.path.join(
		RESULT_PATH, f'SAUCIE-iter{iter_num}-imputed{imputed}-cell_norm{cell_norm}-{lambda_c}-{lambda_d}', data_name)
	os.makedirs(result_folder, exist_ok=True)

	print(f'======================================\n{data_name}: repeat = {repeat_id}')
	ret_list = []
	saucie = SAUCIE(X.shape[1], lambda_c=lambda_c, lambda_d=lambda_d, layers=layers, limit_gpu_fraction=1.0)
	load = Loader(X, shuffle=True)
	saucie.train(load, steps=iter_num)
	num_clusters, y_pred = saucie.get_clusters(load, binmin=0, max_clusters=999999)
	y_pred = y_pred.astype(np.int32)
	assert y_pred.shape==y_true.shape
	print(y_true[:20], y_pred[:20])
	metric_dict = {'ARI':ari(y_true, y_pred), 'NMI':nmi(y_true, y_pred)}
	ret_list.append(metric_dict)
	print(metric_dict)
	json.dump({'SAUCIE': ret_list}, open(os.path.join(result_folder, f'repeat-{repeat_id}.json'), 'w'), indent=2)
	combine_clt_json_to_csv(inpath=result_folder, outpath=os.path.join(result_folder, 'final.csv'))


def main(args):
	repeat = 1
	layers = [int(l_size) for l_size in args.layers.split(',')]

	for data_name in get_all_small_data_names():
		run_single_data(data_name, args.i, imputed=args.imputed, cell_norm=args.cell_norm,
			lambda_c=args.lambda_c, lambda_d=args.lambda_d, iter_num=args.iter_num, layers=layers)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--i', type=int)
	parser.add_argument('--gpu', type=str, default='0')
	parser.add_argument('--imputed', type=int, default=0)
	parser.add_argument('--cell_norm', type=int, default=1)
	parser.add_argument('--lambda_b', type=float, default=0.1)
	parser.add_argument('--lambda_c', type=float, default=0.1)
	parser.add_argument('--lambda_d', type=float, default=0.2)
	parser.add_argument('--iter_num', type=int, default=1000)
	parser.add_argument('--layers', type=str, default='512,256,128,2')
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	# d04 baseline3 get_all_small_data_names
	main(args)




