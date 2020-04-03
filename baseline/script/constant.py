"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import os

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
RESULT_PATH = os.path.join(PROJECT_PATH, 'result')
SCRIPT_PATH = os.path.join(PROJECT_PATH, 'script')
TEMP_PATH = os.path.join(PROJECT_PATH, 'temp')
DATA_PATH = os.path.join(os.path.dirname(PROJECT_PATH), 'sc_cluster', 'data')
PP_DATA_PATH = os.path.join(PROJECT_PATH, 'pp_data')
BASELINE_SRC_PATH = os.path.join(PROJECT_PATH, 'project')

DATA_NAME_TO_FEATURE = {
	'1M_neurons': os.path.join(DATA_PATH, 'raw', '1M_neurons', 'feature.npz'),
	'sc_brain': os.path.join(DATA_PATH, 'raw', 'sc_brain', 'feature_select_gene.npz'),
	'PBMC_68k': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', 'PBMC_68k', 'PBMC_68k.h5'),
	'Shekhar_mouse_retina': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', 'Shekhar_mouse_retina', 'Shekhar_mouse_retina.h5'),
	'10X_PBMC': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', '10X_PBMC', '10X_PBMC.h5'),
	'mouse_bladder_cell': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', 'mouse_bladder_cell', 'mouse_bladder_cell.h5'),
	'mouse_ES_cell': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', 'mouse_ES_cell', 'mouse_ES_cell.h5'),
	'worm_neuron_cell': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', 'worm_neuron_cell', 'worm_neuron_cell.h5'),

	'sim_sparsity_60': os.path.join(DATA_PATH, 'raw', 'sim_sparsity', 'sim_sparsity_60', 'feature.npz'),
	'sim_sparsity_70': os.path.join(DATA_PATH, 'raw', 'sim_sparsity', 'sim_sparsity_70', 'feature.npz'),
	'sim_sparsity_75': os.path.join(DATA_PATH, 'raw', 'sim_sparsity', 'sim_sparsity_75', 'feature.npz'),
	'sim_sparsity_80': os.path.join(DATA_PATH, 'raw', 'sim_sparsity', 'sim_sparsity_80', 'feature.npz'),
	'sim_sparsity_85': os.path.join(DATA_PATH, 'raw', 'sim_sparsity', 'sim_sparsity_85', 'feature.npz'),
	'sim_sparsity_90': os.path.join(DATA_PATH, 'raw', 'sim_sparsity', 'sim_sparsity_90', 'feature.npz'),
	'sim_sparsity_93': os.path.join(DATA_PATH, 'raw', 'sim_sparsity', 'sim_sparsity_93', 'feature.npz'),
	'sim_sparsity_95': os.path.join(DATA_PATH, 'raw', 'sim_sparsity', 'sim_sparsity_95', 'feature.npz'),

	'1M_neurons-1000-samples': os.path.join(DATA_PATH, 'raw', '1M_neurons_samples', 'feature_1000.npz'),
	'1M_neurons-5000-samples': os.path.join(DATA_PATH, 'raw', '1M_neurons_samples', 'feature_5000.npz'),
	'1M_neurons-10000-samples': os.path.join(DATA_PATH, 'raw', '1M_neurons_samples', 'feature_10000.npz'),
	'1M_neurons-50000-samples': os.path.join(DATA_PATH, 'raw', '1M_neurons_samples', 'feature_50000.npz'),
	'1M_neurons-100000-samples': os.path.join(DATA_PATH, 'raw', '1M_neurons_samples', 'feature_100000.npz'),
	'1M_neurons-300000-samples': os.path.join(DATA_PATH, 'raw', '1M_neurons_samples', 'feature_300000.npz'),
	'1M_neurons-500000-samples': os.path.join(DATA_PATH, 'raw', '1M_neurons_samples', 'feature_500000.npz'),
	'1M_neurons-1000000-samples': os.path.join(DATA_PATH, 'raw', '1M_neurons_samples', 'feature_1000000.npz'),
}

DATA_NAME_TO_LABEL = {
	'sc_brain': os.path.join(DATA_PATH, 'raw', 'sc_brain', 'label.npy'),
	'PBMC_68k': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', 'PBMC_68k', 'PBMC_68k.h5'),
	'Shekhar_mouse_retina': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', 'Shekhar_mouse_retina', 'Shekhar_mouse_retina.h5'),
	'10X_PBMC': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', '10X_PBMC', '10X_PBMC.h5'),
	'mouse_bladder_cell': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', 'mouse_bladder_cell', 'mouse_bladder_cell.h5'),
	'mouse_ES_cell': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', 'mouse_ES_cell', 'mouse_ES_cell.h5'),
	'worm_neuron_cell': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', 'worm_neuron_cell', 'worm_neuron_cell.h5'),

	'sim_sparsity_60': os.path.join(DATA_PATH, 'raw', 'sim_sparsity', 'sim_sparsity_60', 'label.npy'),
	'sim_sparsity_70': os.path.join(DATA_PATH, 'raw', 'sim_sparsity', 'sim_sparsity_70', 'label.npy'),
	'sim_sparsity_75': os.path.join(DATA_PATH, 'raw', 'sim_sparsity', 'sim_sparsity_75', 'label.npy'),
	'sim_sparsity_80': os.path.join(DATA_PATH, 'raw', 'sim_sparsity', 'sim_sparsity_80', 'label.npy'),
	'sim_sparsity_85': os.path.join(DATA_PATH, 'raw', 'sim_sparsity', 'sim_sparsity_85', 'label.npy'),
	'sim_sparsity_90': os.path.join(DATA_PATH, 'raw', 'sim_sparsity', 'sim_sparsity_90', 'label.npy'),
	'sim_sparsity_93': os.path.join(DATA_PATH, 'raw', 'sim_sparsity', 'sim_sparsity_93', 'label.npy'),
	'sim_sparsity_95': os.path.join(DATA_PATH, 'raw', 'sim_sparsity', 'sim_sparsity_95', 'label.npy'),
}


if __name__ == '__main__':
	print(PROJECT_PATH)

