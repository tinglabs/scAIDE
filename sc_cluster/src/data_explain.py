"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import os
import numpy as np
import scipy.sparse as sp
from collections import Counter
import json

from constant import RESULT_PATH
from utils_draw import simple_dist_plot
from aide.utils_ import l2_norm, sample_dist_from

EPS = 1e-6

class DataExplainer(object):
	def __init__(self, features, labels=None, data_name=None, save_folder=None, save=False):
		if sp.issparse(features):
			self.explainer = CsrDataExplainer(features, labels, data_name, save_folder, save)
		else:
			self.explainer = AryDataExplainer(features, labels, data_name, save_folder, save)


	def explain(self):
		return self.explainer.explain()


	def draw_feature_nonzero_dist(self):
		return self.explainer.draw_feature_nonzero_dist()


class AryDataExplainer(object):
	def __init__(self, features, labels=None, data_name=None, save_folder=None, save=False):
		"""
		Args:
			features (np.ndarray): (cell_num, gene_num)
			labels (np.ndarray or None): (cell_num,)
		"""
		self.features = features
		self.labels = labels
		self.data_name = data_name
		if save:
			self.SAVE_FOLDER = save_folder or RESULT_PATH+'/data_explain/{}'.format(data_name)
			os.makedirs(self.SAVE_FOLDER, exist_ok=True)
			self.EXPLAIN_DICT_JSON = self.SAVE_FOLDER + '/{}-explain.json'.format(data_name)
			self.FEATURE_NONZERO_DIST_PNG = self.SAVE_FOLDER + '/{}-feature_nonzero_dist.png'.format(data_name)


	def run(self, **kwargs):
		d = self.explain()
		json.dump(d, open(self.EXPLAIN_DICT_JSON, 'w'), indent=2)
		self.draw_feature_nonzero_dist(xlim=kwargs.get('xlim', True))


	def explain(self):
		"""
		Returns:
			dict: {
				'CELL_NUM': int,
				'GENE_NUM': int,
				'FEATURE_QUARTILE': [q0, q25, q50, q75, q100]
				'FEATURE_NON_ZERO_QUARTILE': float
				'LABEL_GROUP_NUM': int,
				'LABEL_GROUP_COUNT': [(label, count), ]
			}
		"""
		X, y = self.features, self.labels
		d = {
			'CELL_NUM': X.shape[0],
			'GENE_NUM': X.shape[1],
			'X_DTYPE': X.dtype.name,
			'Y_DTYPE': y.dtype.name,
			'FEATURE_NON_ZERO_QUARTILE': np.percentile(X, [0, 25, 50, 75, 100]).tolist(),
			'FEATURE_NON_ZERO_RATIO': np.sum(np.abs(X) > EPS) / (X.shape[0] * X.shape[1])
		}
		sample_num = 4000
		d['EUCLIDEAN_DIST_QUATILE'] = np.percentile(sample_dist_from(X, sample_num, 'euclidean'), [0, 25, 50, 75, 100]).tolist()
		d['L2_NORM_QUATILE'] = np.percentile(l2_norm(X), [0, 25, 50, 75, 100]).tolist()
		if y is not None:
			lb_counter = Counter(y)
			d['LABEL_GROUP_COUNT'] = [(int(lb), count) for lb, count in lb_counter.most_common()]
			d['LABEL_GROUP_NUM'] = len(lb_counter)
		return d


	def draw_feature_nonzero_dist(self, figpath=None, xlim=True):
		figpath = figpath or self.FEATURE_NONZERO_DIST_PNG
		data = self.features[np.abs(self.features) > EPS]
		if xlim:
			data = data[data < np.percentile(data, 90)]
		simple_dist_plot(figpath, data, 100, x_label='feature', title='non-zero feature distribution', figsize=(40, 20))


class CsrDataExplainer(object):
	def __init__(self, features, labels=None, data_name=None, save_folder=None, save=False):
		"""
		Args:
			features (csr_matrix): (cell_num, gene_num)
			labels (np.ndarray or None): (cell_num,)
		"""
		self.features = features
		self.labels = labels
		self.data_name = data_name
		if save:
			self.SAVE_FOLDER = save_folder or RESULT_PATH+'/data_explain/{}'.format(data_name)
			os.makedirs(self.SAVE_FOLDER, exist_ok=True)
			self.EXPLAIN_DICT_JSON = self.SAVE_FOLDER + '/{}-explain.json'.format(data_name)
			self.FEATURE_NONZERO_DIST_PNG = self.SAVE_FOLDER + '/{}-feature_nonzero_dist.png'.format(data_name)


	def run(self, **kwargs):
		d = self.explain()
		json.dump(d, open(self.EXPLAIN_DICT_JSON, 'w'), indent=2)
		self.draw_feature_nonzero_dist(xlim=kwargs.get('xlim', True))


	def explain(self):
		"""
		Returns:
			dict: {
				'CELL_NUM': int,
				'GENE_NUM': int,
				'FEATURE_QUARTILE': [q0, q25, q50, q75, q100]
				'FEATURE_NON_ZERO_QUARTILE': float
				'LABEL_GROUP_NUM': int,
				'LABEL_GROUP_COUNT': [(label, count), ]
			}
		"""
		X, y = self.features, self.labels
		d = {
			'CELL_NUM': X.shape[0],
			'GENE_NUM': X.shape[1],
			'X_DTYPE':X.dtype.name,
			'Y_DTYPE':y.dtype.name,
			'FEATURE_NON_ZERO_QUARTILE': np.percentile(X.data, [0, 25, 50, 75, 100]).tolist(),
			'FEATURE_NON_ZERO_RATIO': X.count_nonzero() / (X.shape[0] * X.shape[1])
		}
		sample_num = 4000
		d['EUCLIDEAN_DIST_QUATILE'] = np.percentile(sample_dist_from(X, sample_num, 'euclidean'), [0, 25, 50, 75, 100]).tolist()
		d['L2_NORM_QUATILE'] = np.percentile(l2_norm(X), [0, 25, 50, 75, 100]).tolist()
		if y is not None:
			lb_counter = Counter(y)
			d['LABEL_GROUP_COUNT'] = [(int(lb), count) for lb, count in lb_counter.most_common()]
			d['LABEL_GROUP_NUM'] = len(lb_counter)
		return d


	def draw_feature_nonzero_dist(self, figpath=None, xlim=True):
		figpath = figpath or self.FEATURE_NONZERO_DIST_PNG
		data = self.features.data
		if xlim:
			data = self.features.data[self.features.data < np.percentile(self.features.data, 90)]
		simple_dist_plot(figpath, data, 100, x_label='feature', title='non-zero feature distribution', figsize=(40, 20))



if __name__ == '__main__':
	from reader import get_raw_data, get_all_normal_data_names, get_all_imb_data_names
	# for data_name in get_all_normal_data_names() + get_all_imb_data_names():
	for data_name in ['Shekhar_mouse_retina_IMB', 'mouse_bladder_cell_IMB', 'mouse_ES_cell_IMB']:
		print(data_name)
		features, labels = get_raw_data(data_name)
		explainer = CsrDataExplainer(features, labels, data_name=data_name, save=True)
		explainer.run()


