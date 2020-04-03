"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import os

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
SRC_PATH = os.path.join(PROJECT_PATH, 'src')
RESULT_PATH = os.path.join(PROJECT_PATH, 'result')
MODEL_PATH = os.path.join(RESULT_PATH, 'saved_model')
EMBEDDING_PATH = os.path.join(DATA_PATH, 'embedding_paper')
# EMBEDDING_PATH = os.path.join(DATA_PATH, 'embedding_sparsity_paper')
TEMP_PATH = os.path.join(PROJECT_PATH, 'temp')


DATA_NAME_TO_FOLDER = {
	'1M_neurons': os.path.join(DATA_PATH, 'raw', '1M_neurons'),
	'sc_brain': os.path.join(DATA_PATH, 'raw', 'sc_brain'),
	'PBMC_68k': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', 'PBMC_68k'),
	'Shekhar_mouse_retina': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', 'Shekhar_mouse_retina'),
	'10X_PBMC': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', '10X_PBMC'),
	'mouse_bladder_cell': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', 'mouse_bladder_cell'),
	'mouse_ES_cell': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', 'mouse_ES_cell'),
	'worm_neuron_cell': os.path.join(DATA_PATH, 'raw', 'scDeepCluster', 'worm_neuron_cell'),
	'deng': os.path.join(DATA_PATH, 'raw', 'deng_logged'),
	'llorens': os.path.join(DATA_PATH, 'raw', 'llorens_logged'),
	'mouse_brain_20k': os.path.join(DATA_PATH, 'raw', 'mouse_brain_20k'),
	'Mouse_Bone_Marrow': os.path.join(DATA_PATH, 'raw', 'GSE108097', 'Mouse_Bone_Marrow'),

	'imbalance': os.path.join(DATA_PATH, 'raw', 'imbalance'),
	'sim_sparsity': os.path.join(DATA_PATH, 'raw', 'sim_sparsity'),
}

DENSE_DATA_SET = {'mouse_brain_20k'}


if __name__ == '__main__':
	print(PROJECT_PATH)
