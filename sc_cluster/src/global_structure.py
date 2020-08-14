"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from collections import OrderedDict

from constant import DATA_PATH
from utils_draw import simple_dot_plot, simple_heat_map


def cal_cluster_dist_mat(X, y, cluster_dist='center', label_order=None):
	"""
	Args:
		X (np.ndarray): shape=(n_samples, n_features), dtype=np.float
		y (np.ndarray): shape=(n_samples), dtype=int or str
		cluster_dist: 'ward' | 'center'
	Returns:
		np.ndarray: shape=(n_clusters, n_clusters)
		list of str/int: label order, shape=(n_clusters)
	"""
	if cluster_dist == 'center':
		return cal_cluster_dist_mat_cluster_mean(X, y, label_order)
	elif cluster_dist == 'ward':
		pass
	else:
		raise RuntimeError(f'Unknown cluster distance: {cluster_dist}')


def cal_cluster_dist_mat_cluster_mean(X, y, label_order=None):
	label_to_idx_list = get_label_to_idx_list(y)
	label_order = label_order or list(label_to_idx_list.keys())
	idx_lists = [label_to_idx_list[label] for label in label_order]
	cluster_means = np.vstack([np.mean(X[idx_list], axis=0) for idx_list in idx_lists])
	assert cluster_means.shape == (len(label_order), X.shape[1])
	dist_mat = pairwise_distances(cluster_means, metric='euclidean')
	return dist_mat, label_order


def draw_cluster_dist_mat(X, label_order, figpath, title=None):
	"""
	Args:
		X (np.ndarray): shape=(n_samples, n_samples)
		label_order (list): (n_samples,)
	"""
	simple_heat_map(figpath, X, label_order, label_order, title=title, figsize=(20, 20), colormap='Purples')


def cal_cluster_mean_mds(X, y):
	"""
		Args:
		X (np.ndarray): shape=(n_samples, n_features), dtype=np.float
		y (np.ndarray): shape=(n_samples), dtype=int or str
		figpath (str)
	Returns:
		np.ndarray: shape=(n_clusters, 2)
		list of str/int: label order
	"""
	label_to_idx_list = get_label_to_idx_list(y)
	label_order, idx_lists = zip(*label_to_idx_list.items())
	cluster_means = np.vstack([np.mean(X[idx_list], axis=0) for idx_list in idx_lists])
	assert cluster_means.shape == (len(label_order), X.shape[1])
	embedding = get_mds_embedding(cluster_means, 2)
	print(f'MDS embedding: shape = {embedding.shape}')
	return embedding, label_order


def draw_cluster_mean_mds(X, label_order, figpath):
	simple_dot_plot(
		figpath, X[:, 0], X[:, 1], title='MDS on cluster mean', figsize=(20, 20),
		p_id_to_text={i:label for i, label in enumerate(label_order)})


def get_mds_embedding(X, dim):
	X = X.astype(np.float64).A if sp.issparse(X) else X.astype(np.float64)
	return MDS(n_components=dim).fit_transform(X)


def get_label_to_idx_list(y):
	"""
	Args:
		y (np.ndarray)
	Returns:
		dict: {label: [sample_idx, ...]}
	"""
	d = OrderedDict()
	for i in range(y.shape[0]):
		d.setdefault(y[i], []).append(i)
	return d


if __name__ == '__main__':
	def read_pbmc_Xy():
		X_path = os.path.join(DATA_PATH, 'global_structure', 'pbmc_best_embed.csv')
		y_path = os.path.join(DATA_PATH, 'global_structure', 'pbmc_annotated_labels.csv')
		X = pd.read_csv(X_path).values
		print(f'X: shape = {X.shape}; dtype = {X.dtype}')
		y = np.array(pd.read_csv(y_path).values.flatten().tolist())
		print(f'y: shape = {y.shape}; dtype = {y.dtype}')
		return X, y
	X, y = read_pbmc_Xy()

	# MDS on cluster mean
	# embedding, label_order = cal_cluster_mean_mds(X, y)
	# draw_cluster_mean_mds(embedding, label_order, os.path.join(DATA_PATH, 'global_structure', 'embed_mds.jpg'))

	# Distance Matrix
	label_order = [
		'CD19+ B cells',
		'CD4+ Memory T cells',
		'CD4+/CD25 T Reg cells',
		'CD56+/CD16+ NK cells*',
		'CD56+/CD16- NK cells *',
		'CD8+ Cytotoxic T cells',
		'CD8+/CD45RA+ Naive Cytotoxic T cells',
		'DC1: CD141+ Dendritic Cells *',
		'DC3: Dendritic cells - CD1C_B *',
		'DC4: CD1C-CD141- Dendritic cells *',
		'M1: CD14+ CD16- Classic Monocytes *',
		'pDC Dendritic Cells *',
		'Megakaryocytes',
	]
	cluster_dist = 'center'
	fig_title = 'Distance Matrix on Cluster Mean' if cluster_dist == 'center' else ''
	dist_mat, label_order = cal_cluster_dist_mat(X, y, cluster_dist=cluster_dist, label_order=label_order)
	draw_cluster_dist_mat(dist_mat, label_order,
		os.path.join(DATA_PATH, 'global_structure', f'embed_dist_{cluster_dist}.jpg'), title=fig_title)
