"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import json
import os
import numpy as np
import scipy.sparse as sp
import scanpy.api as sc
from sklearn.utils import check_array

from aide.utils_ import timer, l2_normalize, get_load_func, get_save_func
from reader import get_raw_data
from aide.constant import DATA_MAT, DATA_TFRECORD, NPY_FILE_FORMAT, SPARSE_NPZ_FILE_FORMAT
from constant import DATA_PATH, DENSE_DATA_SET
from data_explain import DataExplainer


class Preprocessor(object):
	def __init__(self, data_name):
		self.data_name = data_name
		self.is_dense = data_name in DENSE_DATA_SET
		self.load_feature = get_load_func(NPY_FILE_FORMAT if self.is_dense else SPARSE_NPZ_FILE_FORMAT)
		self.save_feature = get_save_func(NPY_FILE_FORMAT if self.is_dense else SPARSE_NPZ_FILE_FORMAT)
		self.SAVE_FOLDER = self.get_preprocess_save_folder(data_name)
		os.makedirs(self.SAVE_FOLDER, exist_ok=True)
		self.DATA_INFO_PATH = os.path.join(self.SAVE_FOLDER, 'data_info.json')

		self.NORM_PER_CELL = None


	def has_been_processed(self):
		raise NotImplementedError


	def process(self):
		raise NotImplementedError


	def get_process_features(self):
		raise NotImplementedError


	def get_process_labels(self):
		raise NotImplementedError


	def get_process_data_info(self):
		if not self.has_been_processed():
			self.process()
		return json.load(open(self.DATA_INFO_PATH))


	def get_preprocess_save_folder(self, data_name):
		return os.path.join(DATA_PATH, 'preprocess', data_name)


	def to_adata(self, features, labels=None):
		adata = sc.AnnData(features)
		if labels is not None:
			adata.obs['Group'] = labels
		return adata


	def from_adata(self, adata):
		features, labels = adata.X, None
		if 'Group' in adata.obs:
			labels = adata.obs['Group'].values
		return features, labels


class BasicPreprocessor(Preprocessor):
	"""Handle normal data
	"""
	def __init__(self, data_name):
		super(BasicPreprocessor, self).__init__(data_name)
		if self.is_dense:
			self.FEATURE_PP_PATH = os.path.join(self.SAVE_FOLDER, 'feature_preprocess.npy')
		else:
			self.FEATURE_PP_PATH = os.path.join(self.SAVE_FOLDER, 'feature_preprocess.npz')
		self.LABEL_PP_PATH = os.path.join(self.SAVE_FOLDER, 'label_preprocess.npy')
		self.GENE_MASK_PATH = os.path.join(self.SAVE_FOLDER, 'gene_mask.npy')
		self.CELL_MASK_PATH = os.path.join(self.SAVE_FOLDER, 'cell_mask.npy')


	def has_been_processed(self):
		return os.path.exists(self.FEATURE_PP_PATH)


	@timer
	def process(self):
		features_pp, labels_pp = None, None
		if self.has_been_processed():
			features_pp = self.load_feature(self.FEATURE_PP_PATH)
			if os.path.exists(self.LABEL_PP_PATH):
				labels_pp = np.load(self.LABEL_PP_PATH)
			return features_pp, labels_pp

		features_pp, labels_pp = get_raw_data(self.data_name)
		features_pp, labels_pp = self.process_(features_pp, labels_pp)

		if labels_pp is not None:
			np.save(self.LABEL_PP_PATH, labels_pp)
		self.save_feature(features_pp, self.FEATURE_PP_PATH)

		info_dict = DataExplainer(features_pp, labels_pp).explain()
		info_dict['NORM_PER_CELL'] = self.NORM_PER_CELL
		json.dump(info_dict, open(self.DATA_INFO_PATH, 'w'), indent=2)
		return features_pp, labels_pp


	def get_process_features(self):
		if self.has_been_processed():
			return self.load_feature(self.FEATURE_PP_PATH)
		return self.process()[0]


	def get_process_labels(self):
		if self.has_been_processed():
			return np.load(self.LABEL_PP_PATH) if os.path.exists(self.LABEL_PP_PATH) else None
		return self.process()[1]


	def process_(self, features, labels=None, filter_min_counts=True, same_count_per_cell=True, log1p=True,
				feature_dtype=np.float32, label_dtype=np.int32):
		"""Reference: Tian et.al. Clustering single-cell RNA-seq data with a model-based deep learning approach
		Returns:
			csr_matrix or np.ndarray: (cell_num, gene_num); feature matrix
			np.ndarray: (cell_num, ); label array
		"""
		if filter_min_counts:
			cell_mask_ary, _ = sc.pp.filter_cells(features, min_counts=1, inplace=False)
			np.save(self.CELL_MASK_PATH, cell_mask_ary)
			features = features[cell_mask_ary]
			if labels is not None:
				labels = labels[cell_mask_ary]
			gene_mask_ary, _ = sc.pp.filter_genes(features, min_counts=1, inplace=False)
			np.save(self.GENE_MASK_PATH, gene_mask_ary)
			features = features[:, gene_mask_ary]

		adata = self.to_adata(features, labels)
		if same_count_per_cell:
			sc.pp.normalize_per_cell(adata, counts_per_cell_after=self.NORM_PER_CELL)
		if log1p:
			sc.pp.log1p(adata)
		features, labels = self.from_adata(adata)

		features = check_array(features, accept_sparse="csr", order='C', dtype=[feature_dtype])
		if sp.issparse(features):
			features.sorted_indices()
		if labels is not None:
			labels = check_array(labels, ensure_2d=False, order='C', dtype=[label_dtype])
		return features, labels


class LargeDataPreprocessor(Preprocessor):
	"""Handle extremely large data. Maybe read and write to disk multiple times to avoid being out of memory.
	"""
	def __init__(self, data_name):
		super(LargeDataPreprocessor, self).__init__(data_name)

		postfix = 'npy' if self.is_dense else 'npz'
		self.FEATURE_FILTER_CELL = os.path.join(self.SAVE_FOLDER, f'feature_filter_cell.{postfix}')
		self.LABEL_FILTER_CELL = os.path.join(self.SAVE_FOLDER, f'label_filter_cell.{postfix}')
		self.FEATURE_FILTER_GENE = os.path.join(self.SAVE_FOLDER, f'feature_filter_gene.{postfix}')
		self.FEATURE_NORM = os.path.join(self.SAVE_FOLDER, f'feature_norm.{postfix}')
		self.FEATURE_LOG1P = os.path.join(self.SAVE_FOLDER, f'feature_log1p.{postfix}')
		self.FEATURE_PP_PATH = os.path.join(self.SAVE_FOLDER, f'feature_preprocess.{postfix}')

		self.LABEL_PP_PATH = os.path.join(self.SAVE_FOLDER, 'label_preprocess.npy')

		self.GENE_MASK_PATH = os.path.join(self.SAVE_FOLDER, 'gene_mask.npy')
		self.CELL_MASK_PATH = os.path.join(self.SAVE_FOLDER, 'cell_mask.npy')

		self.HISTORY_JSON = os.path.join(self.SAVE_FOLDER, 'history.json')
		self.history = self.get_history()


	def has_been_processed(self):
		return os.path.exists(self.FEATURE_PP_PATH)


	def get_history(self):
		if os.path.exists(self.HISTORY_JSON):
			return json.load(open(self.HISTORY_JSON))
		return dict()


	def save_history(self, history):
		json.dump(history, open(self.HISTORY_JSON, 'w'), indent=2)


	def process_base(self, key, feature_save_path, label_save_path, handle_func):
		if key in self.history:
			return
		features, labels = handle_func()
		if sp.issparse(features):
			features.sort_indices()
		self.save_feature(feature_save_path, features)
		if labels is not None:
			np.save(label_save_path, labels)
		self.history[key] = True
		self.save_history(self.history)


	def cell_filter(self):
		self.process_base('cell_filter', self.FEATURE_FILTER_CELL, self.LABEL_FILTER_CELL, self.cell_filter_handle_func)


	def cell_filter_handle_func(self):
		features, labels = get_raw_data(self.data_name)
		cell_mask_ary, _ = sc.pp.filter_cells(features, min_counts=1, inplace=False)
		np.save(self.CELL_MASK_PATH, cell_mask_ary)
		features = features[cell_mask_ary]
		if labels is not None:
			labels = labels[cell_mask_ary]
		return features, labels


	def gene_filter(self):
		self.process_base('gene_filter', self.FEATURE_FILTER_GENE, None, self.gene_filter_handle_func)


	def gene_filter_handle_func(self):
		features = self.load_feature(self.FEATURE_FILTER_CELL)
		gene_mask_ary, _ = sc.pp.filter_genes(features, min_counts=1, inplace=False)
		np.save(self.GENE_MASK_PATH, gene_mask_ary)
		features = features[:, gene_mask_ary]
		return features, None


	def normalize_per_cell(self):
		self.process_base('normalize_per_cell', self.FEATURE_NORM, None, self.normalize_per_cell_handle_func)


	def normalize_per_cell_handle_func(self):
		features = self.load_feature(self.FEATURE_FILTER_GENE)
		adata = self.to_adata(features, None)
		sc.pp.normalize_per_cell(adata, counts_per_cell_after=self.NORM_PER_CELL)
		return self.from_adata(adata)


	def log1p(self):
		self.process_base('log1p', self.FEATURE_LOG1P, None, self.log1p_handle_func)


	def log1p_handle_func(self):
		features = self.load_feature(self.FEATURE_NORM)
		adata = self.to_adata(features, None)
		sc.pp.log1p(adata)
		return self.from_adata(adata)


	def final(self):
		self.process_base('final', self.FEATURE_PP_PATH, self.LABEL_PP_PATH, self.final_handle_func)


	def final_handle_func(self):
		features = self.load_feature(self.FEATURE_LOG1P)
		features = check_array(features, accept_sparse="csr", order='C', dtype=[np.float32])
		if sp.issparse(features):
			features.sort_indices()

		labels = None
		if os.path.exists(self.LABEL_FILTER_CELL):
			labels = np.load(self.LABEL_FILTER_CELL)
			print(labels.shape, type(labels), labels.dtype)
			labels = check_array(labels, ensure_2d=False, order='C', dtype=[np.int32])
		info_dict = DataExplainer(features, labels).explain()
		info_dict['NORM_PER_CELL'] = self.NORM_PER_CELL
		json.dump(info_dict, open(self.DATA_INFO_PATH, 'w'), indent=2)
		return features, labels


	def process(self):
		"""
		Returns:
			str: feature file path (.npz, sp.csr_matrix)
			str or None: label file path (.npy, np.ndarray)
		"""
		if not self.has_been_processed():
			print('cell_filter...')
			self.cell_filter()
			print('cell_filter done')

			print('gene_filter...')
			self.gene_filter()
			print('gene_filter done')

			print('normalize_per_cell...')
			self.normalize_per_cell()
			print('normalize_per_cell done')

			print('log1p...')
			self.log1p()
			print('log1p done')

			print('final...')
			self.final()
			print('final done.')

			print('preprocess done!')
		return self.FEATURE_PP_PATH, self.LABEL_PP_PATH if os.path.exists(self.LABEL_PP_PATH) else None


	def get_process_features(self):
		"""
		Returns:
			str: feature file path (.npz, sp.csr_matrix)
		"""
		if self.has_been_processed():
			return self.FEATURE_PP_PATH
		return self.process()[0]


	def get_process_labels(self):
		"""
		Returns:
			str or None: label file path (.npy, np.ndarray)
		"""
		if self.has_been_processed():
			return self.LABEL_PP_PATH if os.path.exists(self.LABEL_PP_PATH) else None
		return self.process()[1]


class TFRecordProcessor(Preprocessor):
	def __init__(self, data_name):
		super(TFRecordProcessor, self).__init__(data_name)
		self.preprocessor = LargeDataPreprocessor(data_name)
		self.TRAIN_FEATURE_SHARDS_FOLDER = os.path.join(self.SAVE_FOLDER, 'train_feature_shards')
		self.PREDICT_FEATURE_TFR = os.path.join(self.SAVE_FOLDER, 'predict_feature.tfrecord')
		self.SHARD_NUM = 10


	def has_been_processed(self):
		return os.path.exists(self.PREDICT_FEATURE_TFR) and os.path.exists(self.TRAIN_FEATURE_SHARDS_FOLDER)


	@timer
	def write_feature_tfrecord(self, features, shuffle):
		from aide.utils_tf import write_csr_to_tfrecord, write_ary_to_tfrecord
		if sp.issparse(features):
			write_csr_to_tfrecord(features, self.PREDICT_FEATURE_TFR, shuffle=shuffle)
		else:
			write_ary_to_tfrecord(features, self.PREDICT_FEATURE_TFR, shuffle=shuffle)


	@timer
	def write_feature_shards_folder(self, features, shuffle):
		from aide.utils_tf import write_csr_shards_to_tfrecord, write_ary_shards_to_tfrecord
		if sp.isspmatrix_csr(features):
			write_csr_shards_to_tfrecord(features, self.TRAIN_FEATURE_SHARDS_FOLDER, shard_num=self.SHARD_NUM, shuffle=shuffle)
		else:
			write_ary_shards_to_tfrecord(features, self.TRAIN_FEATURE_SHARDS_FOLDER, shard_num=self.SHARD_NUM, shuffle=shuffle)


	@timer
	def process(self):
		"""
		Returns:
			tuple: features file, (train_shards_folder, predict_tfrecord);
			str or None: labels file (.npy, np.ndarray)
		"""
		return self.get_process_features(), self.preprocessor.get_process_labels()


	@timer
	def get_process_features(self):
		train_exist = os.path.exists(self.TRAIN_FEATURE_SHARDS_FOLDER)
		pred_exist = os.path.exists(self.PREDICT_FEATURE_TFR)
		if not (train_exist and pred_exist):
			features = sp.load_npz(self.preprocessor.get_process_features())
			self.write_feature_shards_folder(features, shuffle=True)  # train
			self.write_feature_tfrecord(features, shuffle=False)  # predict
		return self.TRAIN_FEATURE_SHARDS_FOLDER, self.PREDICT_FEATURE_TFR


	def get_process_labels(self):
		return self.preprocessor.get_process_labels()


def get_process_data(data_name, data_type=DATA_MAT, **kwargs):
	"""
	Returns:
		sp.csr_matrix or str: features
		np.ndarray or None: labels
	"""
	if data_type == DATA_TFRECORD:
		return TFRecordProcessor(data_name).process()
	elif data_type == DATA_MAT:
		return BasicPreprocessor(data_name).process()
	else:
		raise RuntimeError('Unknown ret_type: {}'.format(data_type))


def get_process_features(data_name, data_type=DATA_MAT, **kwargs):
	"""
	Returns:
		sp.csr_matrix or str
	"""
	if data_type == DATA_TFRECORD:
		return TFRecordProcessor(data_name).get_process_features()
	elif data_type == DATA_MAT:
		return BasicPreprocessor(data_name).get_process_features()
	else:
		raise RuntimeError('Unknown ret_type: {}'.format(data_type))


def get_process_labels(data_name, data_type=DATA_MAT, **kwargs):
	"""
	Returns:
		np.ndarray or str or None
	"""
	if data_type == DATA_TFRECORD:
		return TFRecordProcessor(data_name).get_process_labels()
	elif data_type == DATA_MAT:
		return BasicPreprocessor(data_name).get_process_labels()
	else:
		raise RuntimeError('Unknown ret_type: {}'.format(data_type))


def get_process_data_info(data_name, data_type=DATA_MAT, **kwargs):
	"""
	Returns:
		dict
	"""
	if data_type == DATA_TFRECORD:
		return TFRecordProcessor(data_name).get_process_data_info()
	elif data_type == DATA_MAT:
		return BasicPreprocessor(data_name).get_process_data_info()
	else:
		raise RuntimeError('Unknown ret_type: {}'.format(data_type))


if __name__ == '__main__':
	from reader import get_all_normal_data_names, get_all_extreme_large_data_names

	for data_name in get_all_normal_data_names():
		print('{}-----------------------------'.format(data_name))
		features, labels = get_process_data(data_name)
		print('features:', type(features), features.shape, features.dtype)
		if labels is not None:
			print('labels:', type(labels), labels.shape, labels.dtype)

	# for data_name in get_all_extreme_large_data_names():
	# 	print('{}-----------------------------'.format(data_name))
	# 	(feature_shard_folder, feature_tfrecord), labels_npy = get_process_data(data_name, data_type=DATA_TFRECORD)
	# 	print((feature_shard_folder, feature_tfrecord), labels_npy)


