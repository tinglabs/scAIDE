"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import itertools
import os
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from aide.utils_ import get_load_func, get_save_func, timer


def load_save(attrPath, file_format):
	def outer_wrapper(func):
		def wrapper(cls, *args, **kwargs):
			path = getattr(cls, attrPath)
			if os.path.exists(path):
				load_func = get_load_func(file_format)
				coll = load_func(path)
				return coll
			coll = func(cls, *args, **kwargs)
			saveFunc = get_save_func(file_format)
			saveFunc(coll, path)
			return coll
		return wrapper
	return outer_wrapper


def dict_value_to_list(d):
	ret = {}
	for k, v in d.items():
		if isinstance(v, list) or isinstance(v, tuple):
			ret[k] = v
		else:
			ret[k] = [v]
	return ret


def unzip_dict(d):
	"""
	Args:
		d (dict): e.g. {
			k1: [v1, v2],
			k2: v3,
			k3: [v4, v5]
		}
	Returns:
		list: e.g. [
			{k1: v1, k2: v3, k3: v4},
			{k1: v1, k2: v3, k3: v5},
			{k1: v2, k2: v3, k3: v4},
			{k1: v2, k2: v3, k3: v5}
		]
	"""
	if len(d) == 0:
		return []
	d = dict_value_to_list(d)
	ret_list = []
	k_list, v_lists = zip(*d.items())
	for v_list in itertools.product(*v_lists):
		ret_list.append({k: v for k, v in zip(k_list, v_list)})
	return ret_list


@timer
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


if __name__ == '__main__':
	print(unzip_dict({'k1': ['v1', 'v2'], 'k2': 'v3', 'k3': ['v4', 'v5']}))

