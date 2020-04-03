"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import re
import json
import pandas as pd
import os
import numpy as np
from scipy.sparse import csr_matrix, issparse
import scipy.sparse as sp
import loompy as lp
import h5py
import scanpy
import datatable
from collections import Counter

from utils_ import load_save
from constant import DATA_PATH, DATA_NAME_TO_FOLDER
from aide.utils_ import check_return
from aide.constant import JSON_FILE_FORMAT, SPARSE_NPZ_FILE_FORMAT, NPY_FILE_FORMAT


class Reader(object):
	def __init__(self):
		pass


	def labels_str_to_int(self, lb_str_ary):
		"""
		Args:
			lb_str_ary (np.ndarray): (sample_num,)
		Returns:
			np.ndarray: (sample_num,); dtype=np.int32
			dict: {label: int}
		"""
		lb_str_unique, _ = zip(*sorted(Counter(lb_str_ary).items(), key=lambda item:item[1], reverse=True))
		label2id = {lb_str_unique[i]:i for i in range(len(lb_str_unique))}
		lb_int_ary = np.array(list(map(lambda lb_str:label2id[lb_str], lb_str_ary)), dtype=np.int32)
		return lb_int_ary, label2id


############################################################
class LoomReader(Reader):
	def __init__(self, path, save_folder=None):
		super(LoomReader, self).__init__()
		self.path = path
		self.SAVE_FOLDER = save_folder or os.path.split(path)[0]
		self.FEATURE_SPARSE_NPZ = self.SAVE_FOLDER + os.sep + 'feature.npz'
		self.LABEL_NPY = self.SAVE_FOLDER + os.sep + 'label.npy'
		self.LABEL2ID_JSON = self.SAVE_FOLDER + os.sep + 'label2id.json'


	@check_return('data')
	def get_data(self):
		return lp.connect(self.path, 'r')


	@load_save('FEATURE_SPARSE_NPZ', SPARSE_NPZ_FILE_FORMAT)
	def get_raw_features(self):
		"""
		Returns:
			csr_matrix: (cell_num * gene_num)
		"""
		return self.get_data().sparse().tocsc().T


	@load_save('LABEL_NPY', NPY_FILE_FORMAT)
	def get_labels(self):
		"""
		Returns:
			np.ndarray: (cell_num, )
		"""
		label_to_id = self.get_label2id()
		labels = self.get_data().col_attrs.Class  # np.ndarray
		return np.array(list(map(lambda lb: label_to_id[lb], labels)), dtype=np.int32)


	@load_save('LABEL2ID_JSON', JSON_FILE_FORMAT)
	def get_label2id(self):
		"""
		Returns:
			list: (group_num,)
		"""
		label_names = np.unique(self.get_data().col_attrs.Class).tolist()
		return {lb: i for i, lb in enumerate(label_names)}


############################################################
class H5Reader(Reader):
	def __init__(self, path):
		super(H5Reader, self).__init__()
		self.path = path
		self.data = h5py.File(self.path, 'r')


	def get_raw_features(self):
		"""
		Returns:
			csr_matrix: (cell_num * gene_num)
		"""
		return csr_matrix(self.data['X'])


	def get_labels(self):
		"""
		Returns:
			np.ndarray: (cell_num, )
		"""
		return np.array(self.data['Y'])


############################################################
class Neurons1MReader(Reader):
	def __init__(self, path, save_folder=None):
		super(Neurons1MReader, self).__init__()
		self.path = path
		self.data = scanpy.read_10x_h5(path)
		self.SAVE_FOLDER = save_folder or os.path.split(path)[0]
		self.FEATURE_SPARSE_NPZ = self.SAVE_FOLDER + os.sep + 'feature.npz'


	@load_save('FEATURE_SPARSE_NPZ', SPARSE_NPZ_FILE_FORMAT)
	def get_raw_features(self):
		return self.data.X


	def get_labels(self):
		return None


############################################################
class ScBrainReader(LoomReader):
	def __init__(self, select_genes=True):
		super(ScBrainReader, self).__init__(DATA_NAME_TO_FOLDER['sc_brain'] + os.sep + 'l5_all.loom')
		self.select_genes = select_genes
		self.FEATURE_SELECT_GENE_SPARSE_NPZ = self.SAVE_FOLDER + os.sep + 'feature_select_gene.npz'


	@load_save('FEATURE_SELECT_GENE_SPARSE_NPZ', SPARSE_NPZ_FILE_FORMAT)
	def get_raw_features(self):
		if self.select_genes:
			return self.get_select_genes_features() # (160796, 20803)
		return super(ScBrainReader, self).get_raw_features()    # (160796, 27998)


	@load_save('FEATURE_SELECT_GENE_SPARSE_NPZ', SPARSE_NPZ_FILE_FORMAT)
	def get_select_genes_features(self):
		m = super(ScBrainReader, self).get_raw_features()   # (cell_num, gene_num)
		idx = np.where(self.get_data().row_attrs._Valid == 1)[0]
		return m[:, idx]


############################################################
class MouseBrain20kReader(Reader):
	def __init__(self, folder_path):
		super(MouseBrain20kReader, self).__init__()
		self.SAVE_FOLDER = folder_path
		self.feature_path = os.path.join(self.SAVE_FOLDER, 'expression.txt')
		self.label_path = os.path.join(self.SAVE_FOLDER, 'meta.txt')
		self.FEATURE_NPY = self.SAVE_FOLDER + os.sep + 'feature.npy'
		self.LABEL_NPY = self.SAVE_FOLDER + os.sep + 'label.npy'
		self.LABEL2ID_JSON = self.SAVE_FOLDER + os.sep + 'label2id.json'


	@load_save('FEATURE_NPY', NPY_FILE_FORMAT)
	def get_raw_features(self):
		pd_data = datatable.fread(self.feature_path).to_pandas()
		pd_data = pd_data.drop(['GENE'], axis=1)
		return pd_data.values.T


	@load_save('LABEL_NPY', NPY_FILE_FORMAT)
	def get_labels(self):
		pd_labels = datatable.fread(self.label_path).to_pandas()
		lb_str_ary = pd_labels['All Cell Clusters'][1:].values  # (cell_num,)
		lb_int_ary, label2id = self.labels_str_to_int(lb_str_ary)
		json.dump(label2id, open(self.LABEL2ID_JSON, 'w'), indent=2)
		return lb_int_ary


############################################################
class CsvReader(Reader):
	def __init__(self, folder_path):
		super(CsvReader, self).__init__()
		self.SAVE_FOLDER = folder_path
		self.feature_path = os.path.join(self.SAVE_FOLDER, 'gene_cell_mat.csv')
		self.label_path = os.path.join(self.SAVE_FOLDER, 'label.csv')
		self.FEATURE_SPARSE_NPZ = self.SAVE_FOLDER + os.sep + 'feature.npz'
		self.LABEL_NPY = self.SAVE_FOLDER + os.sep + 'label.npy'
		self.LABEL2ID_JSON = self.SAVE_FOLDER + os.sep + 'label2id.json'


	@load_save('FEATURE_SPARSE_NPZ', SPARSE_NPZ_FILE_FORMAT)
	def get_raw_features(self):
		"""
		Returns:
			csr_matrix or np.ndarray: (cell_num * gene_num)
		"""
		ary = pd.read_csv(self.feature_path, header=0, index_col=0).values.T  # (cell_num, gene_num)
		return sp.csr_matrix(ary)


	@load_save('LABEL_NPY', NPY_FILE_FORMAT)
	def get_labels(self):
		"""
		Returns:
			np.ndarray: (cell_num, )
		"""
		labels = pd.read_csv(self.label_path, header=0, index_col=0).values.flatten()
		if isinstance(labels[0], str):
			labels, label2id = self.labels_str_to_int(labels)
			json.dump(label2id, open(self.LABEL2ID_JSON, 'w'), indent=2)
		return labels


############################################################
class SimDropoutReader(CsvReader):
	def __init__(self, folder_path):
		super(SimDropoutReader, self).__init__(folder_path)


	@load_save('LABEL_NPY', NPY_FILE_FORMAT)
	def get_labels(self):
		df = pd.read_csv(self.label_path, index_col=0)
		lb_strs = df.values.flatten()
		group_num = len(np.unique(lb_strs))
		lb2id = {f'Group{i}': i for i in range(1, group_num + 1)}
		json.dump(lb2id, open(self.LABEL2ID_JSON, 'w'), indent=2)
		return np.array([lb2id[lb] for lb in lb_strs], dtype=np.int32)


def get_raw_data_(name):
	if name == '1M_neurons':
		reader = Neurons1MReader(DATA_NAME_TO_FOLDER['1M_neurons'] + os.sep + '1M_neurons_filtered_gene_bc_matrices_h5.h5')
	elif name == 'sc_brain':
		reader = ScBrainReader(select_genes=True)
	elif name == 'PBMC_68k':
		reader = H5Reader(DATA_NAME_TO_FOLDER['PBMC_68k'] + os.sep + 'PBMC_68k.h5')
	elif name == 'Shekhar_mouse_retina':
		reader = H5Reader(DATA_NAME_TO_FOLDER['Shekhar_mouse_retina'] + os.sep + 'Shekhar_mouse_retina.h5')
	elif name == '10X_PBMC':
		reader = H5Reader(DATA_NAME_TO_FOLDER['10X_PBMC'] + os.sep + '10X_PBMC.h5')
	elif name == 'mouse_bladder_cell':
		reader = H5Reader(DATA_NAME_TO_FOLDER['mouse_bladder_cell'] + os.sep + 'mouse_bladder_cell.h5')
	elif name == 'mouse_ES_cell':
		reader = H5Reader(DATA_NAME_TO_FOLDER['mouse_ES_cell'] + os.sep + 'mouse_ES_cell.h5')
	elif name == 'worm_neuron_cell':
		reader = H5Reader(DATA_NAME_TO_FOLDER['worm_neuron_cell'] + os.sep + 'worm_neuron_cell.h5')
	elif name == 'deng':
		reader = CsvReader(DATA_NAME_TO_FOLDER['deng'])
	elif name == 'llorens':
		reader = CsvReader(DATA_NAME_TO_FOLDER['llorens'])
	else:
		raise RuntimeError('Wrong data name: {}'.format(name))
	return reader.get_raw_features(), reader.get_labels()


def gen_imb_data(X, y_true, keep_group=2, satellite_sample=500):
	y_unique, y_count = zip(*sorted(Counter(y_true).items(), key=lambda item: item[1], reverse=True))
	new_X, new_y_true = [], []
	for i in range(keep_group):
		idx = np.where(y_true == y_unique[i])[0]
		new_X.append(X[idx]); new_y_true.append(y_true[idx])
	for i in range(keep_group, len(y_unique)):
		idx = np.where(y_true == y_unique[i])[0]
		idx = np.random.choice(idx, min(len(idx), satellite_sample), replace=False)
		new_X.append(X[idx]); new_y_true.append(y_true[idx])
	new_X = sp.vstack(new_X) if issparse(X) else np.vstack(new_X)
	new_y_true = np.hstack(new_y_true)
	return new_X, new_y_true


def get_imb_raw_data_(imb_name):
	origin_name = imb_name[:-4]
	save_folder = os.path.join(DATA_NAME_TO_FOLDER['imbalance'], imb_name)
	os.makedirs(save_folder, exist_ok=True)
	feature_path = save_folder + os.sep + 'features.npz'
	labels_path = save_folder + os.sep + 'labels.npy'

	if os.path.exists(feature_path) and os.path.exists(labels_path):
		return sp.load_npz(feature_path), np.load(labels_path)
	features, labels = get_raw_data_(origin_name)
	features, labels = gen_imb_data(features, labels)
	sp.save_npz(feature_path, features)
	np.save(labels_path, labels)
	return features, labels


def get_sim_dropout_raw_data_(data_name):
	folder_path = os.path.join(DATA_NAME_TO_FOLDER['sim_sparsity'], data_name)
	reader = SimDropoutReader(folder_path)
	return reader.get_raw_features(), reader.get_labels()


def gen_sample_data(data_name, n_samples_list):
	save_folder = DATA_NAME_TO_FOLDER[data_name] + '_samples'
	os.makedirs(save_folder, exist_ok=True)
	X, y = get_raw_data(data_name)
	for n_samples in n_samples_list:
		print('Generating samples {}'.format(n_samples))
		x_npz_path = os.path.join(save_folder, 'feature_{}.npz'.format(n_samples))
		y_npy_path = os.path.join(save_folder, 'label_{}.npy'.format(n_samples))
		sample_ranks = np.random.choice(X.shape[0], n_samples, replace=False)
		sp.save_npz(x_npz_path, X[sample_ranks])
		if y is not None:
			np.save(y_npy_path, y[sample_ranks])


def get_sample_data(data_name):
	match_obj = re.match('(.*)-(\d+)-samples', data_name)
	raw_data_name, n_samples = match_obj.group(1), int(match_obj.group(2))
	save_folder = DATA_NAME_TO_FOLDER[raw_data_name] + '_samples'
	x_npz_path = os.path.join(save_folder, 'feature_{}.npz'.format(n_samples))
	y_npy_path = os.path.join(save_folder, 'label_{}.npy'.format(n_samples))
	if not os.path.exists(x_npz_path):
		gen_sample_data(raw_data_name, [n_samples])
	X = sp.load_npz(x_npz_path)
	y = np.load(y_npy_path) if os.path.exists(y_npy_path) else None
	return X, y


def get_raw_data(name):
	"""
	Returns:
		csr_matrix: (cell_num * gene_num); features
		np.ndarray or None: (cell_num, ); labels
	"""
	if name.endswith('IMB'):
		return get_imb_raw_data_(name)
	if name.endswith('samples'):
		return get_sample_data(name)
	if name.startswith('sim_sparsity'):
		return get_sim_dropout_raw_data_(name)
	return get_raw_data_(name)


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


def get_all_imb_data_names():
	return [
		'sc_brain_IMB',
		'PBMC_68k_IMB',
		'10X_PBMC_IMB',
		'worm_neuron_cell_IMB',
	]


def get_all_sim_dropout_data_names(sparsity_list=None):
	sparsity_list = sparsity_list or [60, 70, 75, 80, 85, 90, 93, 95, 97]
	return [f'sim_sparsity_{i}' for i in sparsity_list]


def get_all_sample_data_names(data_name, n_samples_list):
	return [f'{data_name}-{n_samples}-samples' for n_samples in n_samples_list]


def is_extreme_large(data_name):
	return data_name in get_all_extreme_large_data_names()


if __name__ == '__main__':
	# get_raw_data('Mouse_Bone_Marrow')
	gen_sample_data('1M_neurons', [1000, 5000, 10000, 50000, 100000, 500000, 1000000])



