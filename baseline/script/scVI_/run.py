"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

# scVI (downloaded in 2019.11.3): https://github.com/YosefLab/scVI

import json
import pandas as pd
from time import process_time, time
import os
import numpy as np
import argparse

from scvi.dataset import CsvDataset, AnnDatasetFromAnnData
from scvi.models import VAE
from scvi.inference import UnsupervisedTrainer

from script.utils_ import combine_clt_json_to_csv, get_data, get_features_csv, get_labels, run_kmeans_clt, to_adata, combine_metric_dicts
from script.utils_ import get_all_small_data_names, get_all_middle_data_names, get_all_large_data_names, get_all_sim_sparsity_data_names, get_all_sample_data_names
from script.constant import RESULT_PATH, TEMP_PATH


def run_single_data(data_name, format='csv', repeat=5, clt_repeat=10, n_latent=10, n_hidden=128, n_layers=1, n_epochs=20):
	data_save_path = os.path.join(TEMP_PATH, 'scVI', data_name); os.makedirs(data_save_path, exist_ok=True)
	if format == 'csv':
		feature_csv = get_features_csv(data_name, cell_gene=False)
		y_true = get_labels(data_name)
		gene_dataset = CsvDataset(feature_csv, save_path=data_save_path)
	elif format == 'ann':
		X, y_true = get_data(data_name)
		gene_dataset = AnnDatasetFromAnnData(to_adata(X, y_true))
	else:
		raise RuntimeError('Unknown format: {}'.format(format))
	result_folder = os.path.join(RESULT_PATH,
		f'scVI-n_latent{n_latent}-{n_hidden}-{n_layers}-{n_epochs}', data_name)
	reduced_folder = os.path.join(result_folder, 'reduced')

	time_result_folder = os.path.join(result_folder, 'time'); os.makedirs(time_result_folder, exist_ok=True)

	# FIXME: add batch info
	for repeat_id in range(repeat):
		print(f'======================================\n{data_name}-{n_latent}: repeat = {repeat_id}')

		cpu_t, real_t = process_time(), time()

		vae = VAE(gene_dataset.nb_genes, n_batch=0, n_latent=n_latent, n_hidden=n_hidden, n_layers=n_layers)
		trainer = UnsupervisedTrainer(vae, gene_dataset, frequency=1)
		trainer.train(n_epochs=n_epochs)
		full = trainer.create_posterior(trainer.model, gene_dataset, indices=np.arange(len(gene_dataset)))
		h, batch_indices, labels = full.sequential().get_latent()
		# # assert h.shape == (y_true.shape[0], n_latent)

		embedding_cpu_time, embedding_real_time = process_time() - cpu_t, time() - real_t
		print('EMBEDDING_CPU_TIME:', embedding_cpu_time, '; EMBEDDING_REAL_TIME:', embedding_real_time)
		time_dict = {'EMBEDDING_CPU_TIME':embedding_cpu_time, 'EMBEDDING_REAL_TIME':embedding_real_time}
		json.dump(time_dict, open(os.path.join(time_result_folder, f'embedding-{repeat_id}.json'), 'w'), indent=2)

		# print('runing scvi_hidden clustering...')
		# run_kmeans_clt(h, y_true, os.path.join(reduced_folder, f'repeat-{repeat_id}.json'), clt_repeat=clt_repeat)

	# combine_clt_json_to_csv(inpath=reduced_folder, outpath=os.path.join(reduced_folder, 'final.csv'))

	dlist = [json.load(open(os.path.join(time_result_folder, 'embedding-{}.json'.format(i)))) for i in range(repeat)]
	final_result = combine_metric_dicts(dlist)
	pd.DataFrame([final_result]).to_csv(
		time_result_folder + '/final_results.csv', index=False,
		columns=['EMBEDDING_CPU_TIME', 'EMBEDDING_REAL_TIME'])


def main(args):
	# data_names = get_all_sim_sparsity_data_names()
	data_names = get_all_sample_data_names('1M_neurons', [100000])
	repeat = 1
	clt_repeat = 10
	n_epochs = args.n_epochs

	n_latent = args.n_latent
	if n_latent == 10:  # default setting: [128, 10]
		n_hidden = 128
		n_layers = 1
	elif n_latent == 256:   # [512, 512, 256]
		n_hidden = 512
		n_layers = 2
	else:
		assert False

	for data_name in data_names:
		run_single_data(data_name, args.format, repeat, clt_repeat, n_latent=n_latent, n_hidden=n_hidden, n_layers=n_layers, n_epochs=n_epochs)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=str, default='0')
	parser.add_argument('--n_latent', type=int, default=10)
	parser.add_argument('--format', type=str, default='csv')
	parser.add_argument('--n_epochs', type=int, default=400) # recommended by tutorial of scVI: https://nbviewer.jupyter.org/github/YosefLab/scVI/blob/master/tests/notebooks/basic_tutorial.ipynb
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
	# d04 baseline n_latent 10 gpu 2 large
	# d04 baseline2 n_latent 256 gpu 3 middle
	main(args)




