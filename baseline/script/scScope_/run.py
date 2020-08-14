"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

# scScope (downloaded on 2019.11.3): https://github.com/AltschulerWu-Lab/scScope
# Note: The source code of scScope is modified to fix bugs (marked with 'HY modified')

import os, sys
from tqdm import tqdm
from time import process_time, time
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scanpy.api as sc
import phenograph
import json
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
import argparse

from script.utils_ import get_data, to_adata, from_adata, combine_clt_json_to_csv, run_kmeans_clt, timer
from script.utils_ import combine_metric_dicts
from script.constant import RESULT_PATH, TEMP_PATH, BASELINE_SRC_PATH

sys.path.append(BASELINE_SRC_PATH + os.sep + 'scScope' + os.sep + 'scscope')


def select_genes_with_dispersion(X, gene_keep=1000, eps=1e-6, step=1000):
	if sp.issparse(X):
		if not sp.isspmatrix_csc(X):
			X = X.tocsc()
		mean_ary = np.hstack([X[:, i:i+step].mean(axis=0).A1 for i in tqdm(range(0, X.shape[1], step))])
		var_ary = np.hstack([X[:, i:i+step].A.var(axis=0) for i in tqdm(range(0, X.shape[1], step))])
		disp = var_ary / mean_ary  # (n_features,)
		assert np.isnan(disp).sum() == 0 and np.isinf(disp).sum() == 0
		assert mean_ary.shape[0] == X.shape[1] and var_ary.shape[0] == X.shape[1]
		col_idx_ary = np.argsort(disp)[-gene_keep:]
		X = X[:, col_idx_ary]
		return X.tocsr()
	else:
		var_ary = np.var(X+eps, axis=0) # (n_features,)
		mean_ary = np.mean(X+eps, axis=0)   # (n_features,)
		disp = var_ary / mean_ary   # (n_features,)
		assert np.isnan(disp).sum() == 0 and np.isinf(disp).sum() == 0
		assert mean_ary.shape[0] == X.shape[1] and var_ary.shape[0] == X.shape[1]
		col_idx_ary = np.argsort(disp)[-gene_keep:]
		X = X[:, col_idx_ary]
		return X

@timer
def get_data_sc_scope(data_name, gene_keep=1000, log1p=False, scanpy_select=False):
	data_save_folder = os.path.join(TEMP_PATH, 'scScope', 'data', f'{data_name}-gene{gene_keep}-log{log1p}-scanpy_select{scanpy_select}')
	os.makedirs(data_save_folder, exist_ok=True)
	x_npz = os.path.join(data_save_folder, 'feature_preprocess.npz')
	y_npy = os.path.join(data_save_folder, 'label_preprocess.npy')

	if os.path.exists(x_npz):
		X = sp.load_npz(x_npz)
		y_true = np.load(y_npy) if os.path.exists(y_npy) else None
	else:
		X, y_true = get_data(data_name)
		adata = to_adata(X, y_true)
		sc.pp.normalize_per_cell(adata)
		if log1p:
			sc.pp.log1p(adata)
		if gene_keep is not None:
			if scanpy_select:
				assert log1p
				sc.pp.highly_variable_genes(adata, n_top_genes=gene_keep)   # FIXME: no use
				X, y_true = from_adata(adata)
			else:
				X, y_true = from_adata(adata)
				X = select_genes_with_dispersion(X, gene_keep=gene_keep)
			if X.shape[1] != gene_keep:
				raise RuntimeError('X.shape: {}'.format(X.shape))
		sp.save_npz(x_npz, X)
		if y_true is not None:
			np.save(y_npy, y_true)
	if sp.issparse(X):
		X = X.toarray()
	return X, y_true


def get_data_save_folder(data_name, latent_dim, gene_keep, log1p, scanpy_select):
	return os.path.join(TEMP_PATH, 'scScope', f'{data_name}-{latent_dim}-{gene_keep}-{log1p}-{scanpy_select}')


def get_batch_npy(save_folder, i):
	return os.path.join(save_folder, f'batch_{i}.npy')


def get_label_npy(save_folder):
	return os.path.join(save_folder, f'label.npy')


def get_data_info_json(save_folder):
	return os.path.join(save_folder, 'data_info.json')


def split_csr(X, y_true, chunk_size, save_folder):
	cell_num, gene_num = X.shape
	file_num = cell_num // chunk_size
	cell_num -= cell_num % chunk_size   # size of every .npy should be the same, or scScope will failed

	sample_ranks = np.random.choice(X.shape[0], cell_num, replace=False)
	np.save(get_label_npy(save_folder), y_true[sample_ranks])
	for i in range(0, file_num):
		b, e = i*chunk_size, (i+1)*chunk_size
		np.save(get_batch_npy(save_folder, i), X[ sample_ranks[b: e] ])
	json.dump(
		{'cell_num': cell_num, 'gene_num': gene_num, 'file_num': file_num, 'chunk_size': chunk_size},
		open(get_data_info_json(save_folder), 'w'), indent=2)


def get_h(save_folder, file_num):
	return np.vstack([np.load(os.path.join(save_folder, f'feature_{i}.npy')) for i in range(file_num)])


def run_single_large_data(data_name, repeat=5, clt_repeat=10, latent_dim=50,
		encoder_layers=[], decoder_layers=[], T=2, chunk_size=1000, kmeans_num=None, cluster_num=None,
		gene_keep=1000, log1p=True, scanpy_select=False):
	from scscope import large_scale_processing as DeepImpute

	data_folder = get_data_save_folder(data_name, latent_dim=latent_dim, gene_keep=gene_keep, log1p=log1p, scanpy_select=scanpy_select)
	os.makedirs(data_folder, exist_ok=True)
	if not os.path.exists(get_batch_npy(data_folder, 0)):
		X, y_true = get_data_sc_scope(data_name, gene_keep=gene_keep, log1p=log1p, scanpy_select=scanpy_select)
		split_csr(X, y_true, chunk_size, data_folder)
		del X, y_true
	data_info = json.load(open(get_data_info_json(data_folder)))
	y_true = np.load(get_label_npy(data_folder))[: data_info['cell_num']]

	result_folder = os.path.join(RESULT_PATH, f'scScope-latent_dim{latent_dim}-gene_keep{gene_keep}-log1p{log1p}-scanpy_sel{scanpy_select}', data_name)
	reduced_ph_folder = os.path.join(result_folder, 'reduced_phenograph'); os.makedirs(reduced_ph_folder, exist_ok=True)
	reduced_kmeans_folder = os.path.join(result_folder, 'reduced_kmeans'); os.makedirs(reduced_kmeans_folder, exist_ok=True)

	for repeat_id in range(repeat):
		print(f'======================================\nscScope-latent_dim{latent_dim}-gene_keep{gene_keep}-log1p{log1p}-scanpy_sel{scanpy_select}: repeat = {repeat_id}')
		model = DeepImpute.train_large(
			data_folder, data_info['file_num'], data_info['chunk_size'], data_info['gene_num'],
			latent_dim, T=T, batch_size=64, max_epoch=100, num_gpus=1,
			epoch_per_check=100, encoder_layers=encoder_layers, decoder_layers=decoder_layers)
		DeepImpute.predict_large(data_folder, data_info['file_num'], model)
		h = get_h(data_folder, data_info['file_num'])
		assert h.shape == (y_true.shape[0], latent_dim*T)

		run_kmeans_clt(h, y_true, os.path.join(reduced_kmeans_folder, f'repeat-{repeat_id}.json'), clt_repeat=clt_repeat)

		ret_list = []
		for i in range(clt_repeat):
			if kmeans_num is None:
				y_pred, _, _ = phenograph.cluster(h)  # for 'Shekhar_mouse_retina'
			else:
				y_pred = DeepImpute.scalable_cluster(h, kmeans_num=kmeans_num, cluster_num=cluster_num)    # for 'PBMC_68k' and 'sc_brain'
			metric_dict = {'ARI':adjusted_rand_score(y_true, y_pred), 'NMI':normalized_mutual_info_score(y_true, y_pred)}
			ret_list.append(metric_dict)
			print('{}: {}'.format(i, metric_dict))
		json.dump({'sc_scope':ret_list}, open(os.path.join(reduced_ph_folder, f'repeat-{repeat_id}.json'), 'w'), indent=2)

	combine_clt_json_to_csv(inpath=reduced_ph_folder, outpath=os.path.join(reduced_ph_folder, 'final.csv'))
	combine_clt_json_to_csv(inpath=reduced_kmeans_folder, outpath=os.path.join(reduced_kmeans_folder, 'final.csv'))


def run_single_data(data_name, repeat=5, clt_repeat=10, latent_dim=50,
		encoder_layers=[], decoder_layers=[], T=2, kmeans_num=None, cluster_num=None,
		gene_keep=1000, log1p=True, scanpy_select=False):
	import scscope as DeepImpute
	X, y_true = get_data_sc_scope(data_name, gene_keep=gene_keep, log1p=log1p, scanpy_select=scanpy_select) # (cell_num, gene_num)

	print('{}: X.shape={}, y_true.shape={}'.format(data_name, X.shape, y_true.shape if y_true is not None else ''))
	result_folder = os.path.join(RESULT_PATH, f'scScope-latent_dim{latent_dim}-gene_keep{gene_keep}-log1p{log1p}-scanpy_sel{scanpy_select}', data_name)
	reduced_ph_folder = os.path.join(result_folder, 'reduced_phenograph'); os.makedirs(reduced_ph_folder, exist_ok=True)
	reduced_kmeans_folder = os.path.join(result_folder, 'reduced_kmeans'); os.makedirs(reduced_kmeans_folder, exist_ok=True)
	imputed_pca_folder = os.path.join(result_folder, 'imputed_pca'); os.makedirs(imputed_pca_folder, exist_ok=True)

	# time_result_folder = os.path.join(result_folder, 'time'); os.makedirs(time_result_folder, exist_ok=True)

	# FIXME: Not use batch info
	for repeat_id in range(repeat):
		print(f'======================================\nscScope-latent_dim{latent_dim}-gene_keep{gene_keep}-log1p{log1p}-scanpy_sel{scanpy_select}: repeat = {repeat_id}')

		# cpu_t, real_t = process_time(), time()

		model = DeepImpute.train(
			X, latent_dim, T=T, batch_size=64, max_epoch=100, num_gpus=1,
			epoch_per_check=100, encoder_layers=encoder_layers, decoder_layers=decoder_layers)
		h, _, _ = DeepImpute.predict(X, model)

		# embedding_cpu_time, embedding_real_time = process_time() - cpu_t, time() - real_t
		# print('EMBEDDING_CPU_TIME:', embedding_cpu_time, '; EMBEDDING_REAL_TIME:', embedding_real_time)
		# time_dict = {'EMBEDDING_CPU_TIME':embedding_cpu_time, 'EMBEDDING_REAL_TIME':embedding_real_time}
		# # json.dump(time_dict, open(os.path.join(time_result_folder, f'embedding-{repeat_id}.json'), 'w'), indent=2)

		assert h.shape == (X.shape[0], latent_dim*T)

		run_kmeans_clt(h, y_true, os.path.join(reduced_kmeans_folder, f'repeat-{repeat_id}.json'), clt_repeat=clt_repeat)

		ret_list = []
		for i in range(clt_repeat):
			if kmeans_num is None:
				y_pred, _, _ = phenograph.cluster(h)
			else:
				y_pred = DeepImpute.scalable_cluster(h, kmeans_num=kmeans_num, cluster_num=cluster_num)  # for 'PBMC_68k' and 'sc_brain'
			metric_dict = {'ARI':adjusted_rand_score(y_true, y_pred), 'NMI':normalized_mutual_info_score(y_true, y_pred)}
			ret_list.append(metric_dict)
			print('{}: {}'.format(i, metric_dict))
		json.dump({'sc_scope':ret_list}, open(os.path.join(reduced_ph_folder, f'repeat-{repeat_id}.json'), 'w'), indent=2)

	combine_clt_json_to_csv(inpath=reduced_ph_folder, outpath=os.path.join(reduced_ph_folder, 'final.csv'))
	combine_clt_json_to_csv(inpath=reduced_kmeans_folder, outpath=os.path.join(reduced_kmeans_folder, 'final.csv'))

	# dlist = [json.load(open(os.path.join(time_result_folder, 'embedding-{}.json'.format(i)))) for i in range(repeat)]
	# final_result = combine_metric_dicts(dlist)
	# pd.DataFrame([final_result]).to_csv(
	# 	time_result_folder + '/final_results.csv', index=False,
	# 	columns=['EMBEDDING_CPU_TIME', 'EMBEDDING_REAL_TIME'])

def main(args):
	from script.utils_ import get_all_small_data_names, get_all_middle_data_names, get_all_large_data_names
	from script.utils_ import get_all_sim_sparsity_data_names, get_all_sample_data_names
	repeat = 5
	clt_repeat = 10

	T = 2  # Note: output hidden size = latent_dim * T
	latent_dim = args.latent_dim
	if latent_dim == 50:
		encoder_layers = []
		decoder_layers = []
	elif latent_dim == 256:
		encoder_layers = [1024, 512]
		decoder_layers = [512, 1024]
	else:
		raise NotImplementedError

	# for data_name in get_all_sim_sparsity_data_names():
	for data_name in ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina']:
		kmeans_num, cluster_num = None, None
		run_single_data(data_name, repeat=repeat, clt_repeat=clt_repeat, T=T,
			latent_dim=latent_dim, encoder_layers=encoder_layers, decoder_layers=decoder_layers,
			kmeans_num=kmeans_num, cluster_num=cluster_num,
			gene_keep=args.gene_keep, log1p=(args.log1p==1), scanpy_select=(args.scanpy_select==1))

	for data_name in ['PBMC_68k', 'sc_brain']:
		if data_name == 'PBMC_68k':
			kmeans_num, cluster_num = 100, 100
		elif data_name == 'sc_brain':
			kmeans_num, cluster_num = 200, 100
		else:
			assert False
		run_single_large_data(data_name, repeat=repeat, clt_repeat=clt_repeat, T=T, latent_dim=latent_dim,
			encoder_layers=encoder_layers, decoder_layers=decoder_layers,
			kmeans_num=kmeans_num, cluster_num=cluster_num,
			gene_keep=args.gene_keep, log1p=(args.log1p==1), scanpy_select=(args.scanpy_select==1))


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=str, default='0')
	parser.add_argument('--latent_dim', type=int, default=50)
	parser.add_argument('--gene_keep', type=int, default=1000)
	parser.add_argument('--log1p', type=int, default=1)
	parser.add_argument('--scanpy_select', type=int, default=0)
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

	# d04 baseline1 50 small
	# d04 baseline2 256 small
	# d04 baseline3 50 large gpu 0
	# d04 baseline4 256 large gpu 1
	main(args)


