"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import pandas as pd
import json
import os
import h5py
import scipy.sparse as sp
import scipy
import numpy as np
import scanpy.api as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from time import time, process_time
from multiprocessing import Pool

from script.constant import DATA_NAME_TO_FEATURE, DATA_NAME_TO_LABEL, PP_DATA_PATH


def timer(func):
	def wrapper(*args, **kwargs):
		print('{0} starts running...'.format(func.__name__))
		startTime = time()
		ret = func(*args, **kwargs)
		print('Function {0} finished. Total time cost: {1} seconds'.format(func.__name__, time()-startTime))
		return ret
	return wrapper


def combine_clt_json_to_csv(inpath, outpath):
	"""combine *.json in folder, *.json: {clt_name: [metric_dict, ...]}
	"""
	if os.path.isfile(inpath):
		assert inpath.endswith('.json')
		results = [json.load(open(inpath))]
	else:
		results = [json.load(open(os.path.join(inpath, p))) for p in os.listdir(inpath) if p.endswith('.json')]
	# write_list: [{CLT_NAME: '', CLUSTER_CPU_TIME: float, 'CLUSTER_REAL_TIME': float, 'ARI': str, 'NMI': str}, ...]
	write_list = combine_clt_json_base(results)
	pd.DataFrame(write_list).to_csv(
		outpath, index=False,
		columns=['CLT_NAME', 'CLUSTER_CPU_TIME', 'CLUSTER_REAL_TIME', 'ARI', 'NMI'])


def combine_clt_json_base(results):
	final_result = []
	clt_names = results[0].keys()
	for clt_name in clt_names:
		metric_dicts = []
		for result in results:
			metric_dicts.extend(result[clt_name])
		d = combine_metric_dicts(metric_dicts)
		d['CLT_NAME'] = clt_name
		final_result.append(d)
	return final_result


def run_kmeans_clt(X, y_true, save_json, clt_repeat=10, parallel=False):
	"""
	Returns:
		dict: {clt_name: [metric_dict, ...], ...}, metric_dict = {metric_name: score, ...}
	"""
	n_clusters = len(np.unique(y_true))
	print('data_size={}, n_clusters={}'.format(X.shape, n_clusters))

	clt_name_to_args = {}
	for init_type in ['k-means++', 'random']:
		n_init_list = [10] if init_type == 'k-means++' else [10]
		for n_init in n_init_list:
			clt_name = 'kmeans ({}; {})'.format(init_type, n_init)
			clt_initializer, clt_kwargs = KMeans, {'n_clusters': n_clusters, 'init': init_type, 'n_init': n_init}
			clt_name_to_args[clt_name] = (clt_initializer, clt_kwargs)
	ret_dict = run_clt_base(clt_name_to_args, X, y_true, clt_repeat, parallel=parallel)
	os.makedirs(os.path.dirname(save_json), exist_ok=True)
	json.dump(ret_dict, open(save_json, 'w'), indent=2)
	return ret_dict


def run_clt_base(clt_name_to_args, embedding, y_true, clt_repeat, parallel=True):
	"""
	Args:
		clt_name_to_args (dict): {clt_name: (clt_initializer, clt_kwargs)}
	Returns:
		dict: {
			clt_name: [metric_dict, ...]
		}
	"""
	ret_dict = {}
	for clt_name, (clt_initializer, clt_kwargs) in clt_name_to_args.items():
		if parallel:
			with Pool(10) as pool:
				dlist = pool.map(get_clt_performance,
					[(embedding, y_true, clt_initializer, clt_kwargs) for i in range(clt_repeat)])
		else:
			dlist = [get_clt_performance(
				(embedding, y_true, clt_initializer, clt_kwargs)) for i in range(clt_repeat)]
		ret_dict[clt_name] = dlist
		print('## {} ##'.format(clt_name), combine_metric_dicts(dlist))
	return ret_dict


def combine_metric_dicts(dlist):
	"""
	Args:
		dlist (list): [{metric_name: score}, ...]
	Returns:
		dict: {metric_name: score_mean (score_std)}
	"""
	d = {}
	for k in dlist[0]:
		score_list = [metric_dict[k] for metric_dict in dlist]
		ave = np.mean(score_list)
		std = np.std(score_list)
		d[k] = '{:.3} ({:.3})'.format(ave, std)
	return d


def get_clt_performance(args):
	"""
	Returns:
		dict: {
			'ARI': float
			'NMI': float
			'CLUSTER_TIME'
		}
	"""
	scipy.random.seed()  # Important
	X, y_true, clt_initializer, clt_kwargs = args

	p_time, t_time = process_time(), time()
	clt = clt_initializer(**clt_kwargs)
	y_pred = clt.fit_predict(X)
	p_time_spend, t_time_spend = process_time() - p_time, time() - t_time

	d = {
		'CLUSTER_REAL_TIME': t_time_spend,
		'CLUSTER_CPU_TIME': p_time_spend,
		'ARI': adjusted_rand_score(y_true, y_pred),
		'NMI': normalized_mutual_info_score(y_true, y_pred),
	}
	print(d)
	return d


def count_non_zero(X, eps=1e-6):
	if sp.issparse(X):
		return X.count_nonzero()
	else:
		return (np.abs(X) > eps).sum()


def get_pp_folder(data_name):
	return os.path.join(PP_DATA_PATH, data_name)


def get_pp_feature_path(data_name, mark='', postfix='npz'):
	pp_folder = get_pp_folder(data_name)
	return os.path.join(pp_folder, f'pp_features_{mark}.{postfix}' if mark else f'pp_features.{postfix}')


def get_pp_label_path(data_name, mark='', postfix='npy'):
	pp_folder = get_pp_folder(data_name)
	return os.path.join(pp_folder, f'pp_labels_{mark}.{postfix}' if mark else f'pp_labels.{postfix}')


def get_data_h5ad(data_name, cell_gene=True):
	h5ad_path = os.path.join(get_pp_folder(data_name), 'data.h5ad')
	if os.path.exists(h5ad_path):
		return h5ad_path
	X, y = get_data(data_name)
	adata = to_adata(X, y)
	adata.write_h5ad(h5ad_path)
	return h5ad_path


def get_data_csv(data_name, cell_gene=True):
	return get_features_csv(data_name, cell_gene), get_labels_csv(data_name)


@timer
def get_features_csv(data_name, cell_gene=True, compression='infer'):
	mark = 'cell_gene' if cell_gene else 'gene_cell'
	postfix = 'csv.gz' if compression == 'gzip' else 'csv'
	csv_path = get_pp_feature_path(data_name, mark=mark, postfix=postfix)
	if not os.path.exists(csv_path):
		X = get_features(data_name)
		if cell_gene:
			pd.DataFrame(X.toarray()).to_csv(csv_path, compression=compression)  # (cell_num, gene_num)
		else:
			pd.DataFrame(X.T.toarray()).to_csv(csv_path)    # (gene_num, cell_num)
	return csv_path


def get_labels_csv(data_name):
	csv_path = get_pp_label_path(data_name, 'csv')
	if not os.path.exists(csv_path):
		y = get_labels(data_name)
		pd.DataFrame(y).to_csv(csv_path)
	return csv_path


def get_data(data_name):
	X, y = None, None
	pp_X_path, pp_y_path = get_pp_feature_path(data_name), get_pp_label_path(data_name)
	if os.path.exists(pp_X_path):
		X = sp.load_npz(pp_X_path)
		if os.path.exists(pp_y_path):
			y = np.load(pp_y_path)
		return X, y

	X, y = get_raw_features(data_name), get_raw_labels(data_name)
	adata = to_adata(X, y)
	sc.pp.filter_cells(adata, min_counts=1)
	sc.pp.filter_genes(adata, min_counts=1)
	X, y = from_adata(adata)
	X.sorted_indices()

	os.makedirs(get_pp_folder(data_name), exist_ok=True)
	sp.save_npz(pp_X_path, X)
	if y is not None:
		np.save(pp_y_path, y)

	return X, y


def get_features(data_name):
	pp_X_path = get_pp_feature_path(data_name)
	if os.path.exists(pp_X_path):
		return sp.load_npz(pp_X_path)
	X, _ = get_data(data_name)
	return X


def get_labels(data_name):
	pp_X_path, pp_y_path = get_pp_feature_path(data_name), get_pp_label_path(data_name)
	if os.path.exists(pp_X_path):
		return np.load(pp_y_path) if os.path.exists(pp_y_path) else None
	_, y = get_data(data_name)
	return y


def get_raw_features(data_name):
	path = DATA_NAME_TO_FEATURE.get(data_name, '')
	if path.endswith('.npz'):
		return sp.load_npz(path).astype(np.float32)
	elif path.endswith('.h5'):
		f = h5py.File(path, 'r')
		return sp.csr_matrix(f['X'], dtype=np.float32)
	else:
		return None


def get_raw_labels(data_name):
	path = DATA_NAME_TO_LABEL.get(data_name, '')
	if path.endswith('.npy'):
		return np.load(path).astype(np.int32)
	elif path.endswith('.h5'):
		f = h5py.File(path, 'r')
		return np.array(f['Y'], dtype=np.int32)
	else:
		return None


def to_adata(features, labels=None):
	adata = sc.AnnData(features)
	if labels is not None:
		adata.obs['Group'] = labels
	return adata


def from_adata(adata):
	features, labels = adata.X, None
	if 'Group' in adata.obs:
		labels = adata.obs['Group'].values
	return features, labels


def get_all_data_names():
	return [
		'1M_neurons',
		'sc_brain',
		'PBMC_68k',
		'Shekhar_mouse_retina',
		'10X_PBMC',
		'mouse_bladder_cell',
		'mouse_ES_cell',
		'worm_neuron_cell',
	]


def get_all_normal_data_names():
	return get_all_small_data_names() + get_all_middle_data_names() + get_all_large_data_names()


def get_all_small_data_names():
	return [
		'10X_PBMC',
		'mouse_bladder_cell',
		'mouse_ES_cell',
		'worm_neuron_cell'
	]


def get_all_middle_data_names():
	return [
		'Shekhar_mouse_retina',
		'PBMC_68k',
	]


def get_all_large_data_names():
	return [
		'sc_brain'
	]


def get_all_extreme_large_data_names():
	return [
		'1M_neurons'
	]


def get_all_sim_sparsity_data_names():
	return [
		'sim_sparsity_60',
		'sim_sparsity_70',
		'sim_sparsity_75',
		'sim_sparsity_80',
		'sim_sparsity_85',
		'sim_sparsity_90',
		'sim_sparsity_93'
	]


def get_all_sample_data_names(data_name, n_samples_list):
	return [f'{data_name}-{n_samples}-samples' for n_samples in n_samples_list]

