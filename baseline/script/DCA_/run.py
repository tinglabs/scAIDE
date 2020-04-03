"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

# dca (downloaded in 2019.11.3): https://github.com/theislab/dca
# Note: the source code of DCA is modified for easier running (marked with 'HY modified')

import os
import scanpy.api as sc
from sklearn.decomposition import PCA
import numpy as np

from dca.__main__ import parse_args
from dca.train import train_with_args

from script.utils_ import get_data, run_kmeans_clt, to_adata
from script.utils_ import combine_clt_json_to_csv
from script.constant import TEMP_PATH, RESULT_PATH, DATA_PATH
from script.utils_ import get_all_small_data_names, get_all_middle_data_names, get_all_sim_sparsity_data_names


def run_single_data(data_name, repeat=5, clt_repeat=10, hidden_size='64,32,64', save_impute=False):
	X, y_true = get_data(data_name)
	x_shape = X.shape
	args = parse_args()
	args.hiddensize = hidden_size
	args.input = to_adata(X.T.toarray())
	result_folder = os.path.join(RESULT_PATH, f'dca-{hidden_size}', data_name)
	reduced_folder = os.path.join(result_folder, 'reduced')
	imputed_pca_folder = os.path.join(result_folder, 'imputed_pca')
	imputed_folder = os.path.join(result_folder, 'imputed')
	impute_save_path = os.path.join(DATA_PATH, 'impute', f'DAC-{hidden_size}', data_name)
	del X

	for repeat_id in range(repeat):
		print(f'======================================\n{data_name}: repeat = {repeat_id}')
		model_folder = os.path.join(TEMP_PATH, f'dca-{hidden_size}', f'{data_name}-{repeat_id}')
		os.makedirs(model_folder, exist_ok=True)
		args.outputdir = model_folder
		adata = train_with_args(args)
		assert adata.X.shape == x_shape and (args.input.X[:, 10].T != adata.X[:10]).any()

		if save_impute:
			os.makedirs(impute_save_path, exist_ok=True)
			np.save(os.path.join(impute_save_path, f'impute-{repeat_id}.npy'), adata.X)

		sc.pp.normalize_per_cell(adata)
		sc.pp.log1p(adata)

		# performance is pool
		# print('running dca_hidden clustering...')
		# run_kmeans_clt(adata.obsm['X_dca'], y_true, os.path.join(reduced_folder, f'repeat-{repeat_id}.json'), clt_repeat=clt_repeat)

		print('=======================================\nrunning dca_imputed_pca clustering...')
		pca_X = PCA(n_components=256).fit_transform(adata.X)
		run_kmeans_clt(pca_X, y_true, os.path.join(imputed_pca_folder, f'repeat-{repeat_id}.json'), clt_repeat=clt_repeat)

		# print('=======================================\nrunning dca_imputed clustering...')
		# run_kmeans_clt(adata.X, y_true, os.path.join(imputed_folder, f'repeat-{repeat_id}.json'), clt_repeat=clt_repeat)

	# combine_clt_json_to_csv(inpath=reduced_folder, outpath=os.path.join(reduced_folder, 'final.csv'))
	combine_clt_json_to_csv(inpath=imputed_pca_folder, outpath=os.path.join(imputed_pca_folder, 'final.csv'))
	# combine_clt_json_to_csv(inpath=imputed_folder, outpath=os.path.join(imputed_folder, 'final.csv'))


def main():
	"""Note: for 'Shekhar_mouse_retina', run with CPU to avoid GPU's memory error;
		for other data in get_all_small_data_names(), run with GPU.
	"""
	repeat = 5
	clt_repeat=10
	hidden_size = '1024,512,256,512,1024' # ''64,32,64''
	for data_name in ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell']:
		run_single_data(data_name, repeat=repeat, clt_repeat=clt_repeat, hidden_size=hidden_size, save_impute=False)


if __name__ == '__main__':
	os.environ["CUDA_VISIBLE_DEVICES"] = '0'
	# d02 baseline2 32 small gpu 0
	# d02 baseline 256 small gpu 1
	main()

