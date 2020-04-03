"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

# MAGIC (downloaded in 2019.11.5): https://github.com/KrishnaswamyLab/magic

import os
import scprep
import pandas as pd
import argparse
import scipy.sparse as sp

import magic

from script.utils_ import get_data, run_kmeans_clt
from script.utils_ import combine_clt_json_to_csv
from script.constant import TEMP_PATH, RESULT_PATH
from script.utils_ import get_all_small_data_names, get_all_middle_data_names, get_all_large_data_names, get_all_sim_sparsity_data_names
from script.utils_ import count_non_zero


def preprocess(X, y):
	df = pd.DataFrame(X.toarray())
	df = scprep.filter.filter_rare_genes(df, min_cells=10)   # recommended by tutorial of MAGIC: https://nbviewer.jupyter.org/github/KrishnaswamyLab/MAGIC/blob/master/python/tutorial_notebooks/bonemarrow_tutorial.ipynb
	df = scprep.normalize.library_size_normalize(df)
	df = scprep.transform.sqrt(df)
	non_zero_rate = count_non_zero(df.values) / (df.shape[0] * df.shape[1])
	if non_zero_rate < 0.3:
		X = sp.csr_matrix(df.values)
	else:
		X = df.values
	assert X.shape[0] == y.shape[0]
	return X, y


def run_single_data(data_name, repeat=5, clt_repeat=10, knn=10, magic_pca=100, n_jobs=12):
	X, y_true = get_data(data_name)
	print('Before preprocess: X.shape = {}'.format(X.shape))
	X, y_true = preprocess(X, y_true)
	print('After preprocess: X.shape = {}'.format(X.shape))
	result_folder = os.path.join(RESULT_PATH, f'MAGIC-knn{knn}-magic_pca{magic_pca}', data_name)

	for repeat_id in range(repeat):
		print(f'======================================\n{data_name}: repeat = {repeat_id}')
		magic_op = magic.MAGIC(knn=knn, n_pca=magic_pca, n_jobs=n_jobs)
		X_imputed_pca = magic_op.fit_transform(X, genes='pca_only')
		assert X_imputed_pca.shape == (X.shape[0], magic_pca)
		run_kmeans_clt(X_imputed_pca, y_true, os.path.join(result_folder, f'repeat-{repeat_id}.json'), clt_repeat=clt_repeat)
	combine_clt_json_to_csv(inpath=result_folder, outpath=os.path.join(result_folder, 'final.csv'))


def main(args):
	repeat = 5
	clt_repeat = 10

	for data_name in ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']:
		run_single_data(data_name, repeat, clt_repeat,
			knn=args.knn, magic_pca=args.magic_pca, n_jobs=args.n_jobs)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--knn', type=int, default=10)
	parser.add_argument('--magic_pca', type=int, default=100)
	parser.add_argument('--n_jobs', type=int, default=12)
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	# d04 baseline5 pca100
	# c01 pca256 PBMC_68k
	# c04 pca256 sc_brain
	main(args)
