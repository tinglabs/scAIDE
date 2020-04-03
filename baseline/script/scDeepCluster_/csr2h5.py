"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import os
import h5py
import scipy.sparse as sp
import numpy as np

from script.utils_ import get_data, get_all_sim_sparsity_data_names, get_all_sample_data_names
from script.constant import PROJECT_PATH

if __name__ == '__main__':
	# SAVE_PATH = os.path.join(PROJECT_PATH, 'project', 'scDeepCluster', 'scRNA_seq_data', 'sim_sparsity')
	# os.makedirs(SAVE_PATH, exist_ok=True)
	# data_names = get_all_sim_sparsity_data_names()

	SAVE_PATH = os.path.join(PROJECT_PATH, 'project', 'scDeepCluster', 'scRNA_seq_data', '1M_neurons_samples')
	os.makedirs(SAVE_PATH, exist_ok=True)
	data_names = get_all_sample_data_names('1M_neurons', [1000, 5000, 10000, 50000, 100000, 300000])

	for data_name in data_names:
		print('handling {}'.format(data_name))
		features, labels = get_data(data_name)

		if sp.isspmatrix_csr(features):
			features = features.toarray()
		h5path = os.path.join(SAVE_PATH, f'{data_name}.h5')
		f = h5py.File(h5path, 'w')
		f.create_dataset('X', data=features)
		if labels is None:
			labels = np.zeros(features.shape[0], dtype=np.int32)
		f.create_dataset('Y', data=labels)
		f.close()


