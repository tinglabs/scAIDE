
import re
import os
import numpy as np

from script.constant import RESULT_PATH
from script.utils_ import get_all_sim_sparsity_data_names

def str_to_float_list(l):
	return list(map(lambda item: float(item), l))


# hidden_256
# REPEAT = 5
# DATA_NAMES = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']
# FOLDER = os.path.join(RESULT_PATH, 'scDeepCluster-256')
# LOG_FILE = os.path.join(FOLDER, 'run_256.log')

# default
# REPEAT = 5
# DATA_NAMES = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']
# FOLDER = os.path.join(RESULT_PATH, 'scDeepCluster-32')
# LOG_FILE = os.path.join(FOLDER, 'run_default.log')

# pretrained
# REPEAT = 5
# DATA_NAMES = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell']
# FOLDER = os.path.join(RESULT_PATH, 'scDeepCluster-paper')
# LOG_FILE = os.path.join(FOLDER, 'run_pretrained.log')

# default sim_sparsity
REPEAT = 5
DATA_NAMES = get_all_sim_sparsity_data_names()
FOLDER = os.path.join(RESULT_PATH, 'scDeepCluster-32')
LOG_FILE = os.path.join(FOLDER, 'run_sim_sparsity_default.log')

LIST_LEN = REPEAT * len(DATA_NAMES)
s = open(LOG_FILE).read()

pretrain_times = re.findall('pretrain time = (\d+) seconds', s)
assert len(pretrain_times) == LIST_LEN
pretrain_times = str_to_float_list(pretrain_times)
# print('pretrain_times: {}\n'.format(pretrain_times))

cluster_times = re.findall('Clustering time: (\d+) seconds', s)
assert len(cluster_times) == LIST_LEN
cluster_times = str_to_float_list(cluster_times)
# print('cluster_times: {}\n'.format(cluster_times))

all_times = [t1+t2 for t1, t2 in zip(cluster_times, cluster_times)]

scores = re.findall('Final: ACC= (\d+\.\d+), NMI= (\d+\.\d+), ARI= (\d+\.\d+)', s)
_, nmi_scores, ari_scores = zip(*scores)
nmi_scores, ari_scores = str_to_float_list(nmi_scores), str_to_float_list(ari_scores)
# print('ari_scores: {}\n'.format(ari_scores))
# print('nmi_scores: {}\n'.format(nmi_scores))

print_order = ['ARI', 'NMI', 'pretrain_times', 'cluster_times', 'all_times']
print_data = {'ARI': ari_scores, 'NMI': nmi_scores, 'pretrain_times': pretrain_times, 'cluster_times': cluster_times, 'all_times': all_times}
for i in range(len(DATA_NAMES)):
	print(DATA_NAMES[i])
	b, e = i*5, (i+1)*5
	for k in print_order:
		data = print_data[k][b: e]
		print('\t{}: mean (std) = {:.3f} ({:.3f}); data={}'.format(k, np.mean(data), np.std(data), data))






