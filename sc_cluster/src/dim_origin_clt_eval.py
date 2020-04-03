"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

# Note: No need to limit the threads number as done in 'dim_reduce_clt_eval.py' because there is no significant reduction of CPU time.

import os
import scipy
import pandas as pd
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import json
from time import time, process_time
import itertools
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import phenograph as pg

from constant import RESULT_PATH
from preprocess import BasicPreprocessor, get_process_data
from reader import get_raw_data
from rp_kmeans import RPKMeans

class DimOriginCltEvaluator(object):
	def __init__(self):
		self.metrics = ['ARI', 'NMI', 'CPU_TIME', 'REAL_TIME']

		self.SAVE_FOLDER = RESULT_PATH + os.sep + 'dim_origin_clt_eval'
		self.RAW_RESULT_FOLDER = self.SAVE_FOLDER + os.sep + 'raw_result'
		self.METRIC_FOLDER = self.SAVE_FOLDER + os.sep + 'metric'
		self.SUMMARY_METRIC_FOLDER = self.SAVE_FOLDER + os.sep + 'summary'

		os.makedirs(self.RAW_RESULT_FOLDER, exist_ok=True)
		os.makedirs(self.METRIC_FOLDER, exist_ok=True)
		os.makedirs(self.SUMMARY_METRIC_FOLDER, exist_ok=True)


	def get_raw_result_save_path(self, data_name, clt_name, repeat):
		path = self.RAW_RESULT_FOLDER + '/{}/{}/repeat-{}-labels.npy'.format(data_name, clt_name, repeat)
		os.makedirs(os.path.split(path)[0], exist_ok=True)
		return path


	def get_metric_save_path(self, data_name, clt_name, repeat):
		path = self.METRIC_FOLDER + '/{}/{}/repeat-{}.json'.format(data_name, clt_name, repeat)
		os.makedirs(os.path.split(path)[0], exist_ok=True)
		return path


	def get_performance(self, y_true, y_pred):
		"""
		Returns:
			dict: {metric_name: score}
		"""
		d = {
			'ARI': adjusted_rand_score(y_true, y_pred),
			'NMI': normalized_mutual_info_score(y_true, y_pred),
		}
		return d


	def cal_metric_wrapper(self, para):
		scipy.random.seed() # Very important, or clt result may be the same
		clt_initializer, clt_kwargs, clt_name, data_name, feature_dtype, repeat_id, save_raw_result = para
		print(clt_name, data_name, repeat_id)

		# get data
		features, y_true = get_raw_data(data_name)
		features, y_true = BasicPreprocessor(data_name).process_(features, y_true, feature_dtype=feature_dtype)

		# run clustering
		p_time, t_time = process_time(), time()
		clt = clt_initializer(**clt_kwargs)
		y_pred = clt.fit_predict(features)
		p_time_spend, t_time_spend = process_time() - p_time, time() - t_time

		metric_dict = self.get_performance(y_true, y_pred)
		metric_dict['CPU_TIME'] = p_time_spend
		metric_dict['REAL_TIME'] = t_time_spend
		metric_dict['DATA_NAME'] = data_name
		metric_dict['MODEL_NAME'] = clt_name
		metric_dict['REPEAT'] = repeat_id

		print(data_name, clt_name, repeat_id, metric_dict)
		metric_json = self.get_metric_save_path(data_name, clt_name, repeat_id)
		json.dump(metric_dict, open(metric_json, 'w'), indent=2)

		if save_raw_result:
			raw_result_npy = self.get_raw_result_save_path(data_name, clt_name, repeat_id)
			np.save(raw_result_npy, y_pred)


	def run(self, arg_lists, cpu=12, save_raw_result=True):
		"""
		Args:
			arg_lists (list): [[
				clt_initializer,
				clt_kwargs,
				clt_name,
				data_name,
				feature_dtype,
				repeat_id,
			], ...]
		"""
		para_list = [arg_list + [save_raw_result] for arg_list in arg_lists]
		if cpu == 1:
			for para in tqdm(para_list):
				self.cal_metric_wrapper(para)
		else:
			with Pool(cpu) as pool:
				for _ in tqdm(pool.imap_unordered(self.cal_metric_wrapper, para_list), total=len(para_list), leave=False):
					pass


	def gen_summary_csv(self, clt_names, data_names, folder=None, repeat_times=1):
		def get_key(metric_name, data_name, clt_name, repeat):
			return '{}-{}-{}-{}'.format(metric_name, data_name, clt_name, repeat)
		folder = folder or self.SUMMARY_METRIC_FOLDER

		d = {}
		for data_name in data_names:
			for clt_name in clt_names:
				for repeat in range(0, repeat_times):
					metric_file = self.get_metric_save_path(data_name, clt_name, repeat)
					if not os.path.exists(metric_file):
						print('warning: {} not exist!'.format(metric_file))
						continue
					metric_dict = json.load(open(metric_file))
					for metric_name in self.metrics:
						d[get_key(metric_name, data_name, clt_name, repeat)] = metric_dict[metric_name]

		for metric_name in self.metrics:
			df_list = []    # row: clt_name col: data_name;
			for clt_name in clt_names:
				row_dict = {}
				for data_name in data_names:
					key_list = [get_key(metric_name, data_name, clt_name, repeat)
						for repeat in range(0, repeat_times)]
					score_list = [d[key] for key in key_list if key in d]
					if len(score_list) == 1:
						row_dict[data_name] = '{:.3f}'.format(score_list[0])
					else:
						row_dict[data_name] = '{:.3f} ({:.3f})'.format(np.mean(score_list), np.std(score_list))
				row_dict['MODEL'] = clt_name
				df_list.append(row_dict)
			csv_path = folder + '/{}.csv'.format(metric_name)
			pd.DataFrame(df_list, columns=['MODEL'] + data_names).to_csv(csv_path, index=False)


class PhenographCluster(object):
	def __init__(self, name=None, **kwargs):
		super(PhenographCluster, self).__init__()
		self.name = name or 'phenograph'
		self.kwargs = kwargs


	def fit_predict(self, X):
		labels, graph, Q = pg.cluster(X, **self.kwargs)
		return labels


def run_phenograph(cpu_use, repeat_times, data_names):
	arg_lists = []
	for data_name, repeat_id in itertools.product(data_names, range(repeat_times)):
		arg_lists.append([
			PhenographCluster,
			{'n_jobs': 1},
			'phenograph',
			data_name,
			np.float64, # Note: If set to np.float32, the performance will be bad
			repeat_id
		])
	evaluator = DimOriginCltEvaluator()
	evaluator.run(arg_lists, cpu=cpu_use)


def run_kmeans(init_list, n_init_list, cpu_use, repeat_times, data_names):
	data_n_cluster = {}
	for data_name in data_names:
		_, labels = get_process_data(data_name)
		data_n_cluster[data_name] = len(np.unique(labels))

	arg_lists = []
	for init, n_init, data_name, repeat_id in itertools.product(
			init_list, n_init_list, data_names, range(repeat_times)):
		arg_lists.append([
			KMeans,
			{'init': init, 'n_init': n_init, 'n_clusters': data_n_cluster[data_name], 'n_jobs':1},
			'kmeans ({}; {})'.format(init, n_init),
			data_name,
			np.float32,
			repeat_id
		])
	evaluator = DimOriginCltEvaluator()
	evaluator.run(arg_lists, cpu=cpu_use)


def run_rp_kmeans(n_init_list, cpu_use, repeat_times, data_names):
	data_n_cluster = {}
	for data_name in data_names:
		_, labels = get_process_data(data_name)
		data_n_cluster[data_name] = len(np.unique(labels))

	arg_lists = []
	for n_init, data_name, repeat_id in itertools.product(n_init_list, data_names, range(repeat_times)):
		arg_lists.append([
			RPKMeans,
			{'n_clusters': data_n_cluster[data_name], 'n_init': n_init},
			'rp kmeans (default; {})'.format(n_init),
			data_name,
			np.float32,
			repeat_id
		])
	evaluator = DimOriginCltEvaluator()
	evaluator.run(arg_lists, cpu=cpu_use)


if __name__ == '__main__':
	from reader import get_all_sim_dropout_data_names
	# example:
	data_names = ['10X_PBMC']
	repeat_times = 10
	cpu_use = 20

	run_phenograph(cpu_use=cpu_use, repeat_times=repeat_times, data_names=data_names)
	run_kmeans(init_list = ['random', 'k-means++'], n_init_list = [1, 10],
		cpu_use=cpu_use, repeat_times=repeat_times,  data_names=data_names)
	run_rp_kmeans(n_init_list=[1], cpu_use=cpu_use, repeat_times=repeat_times, data_names=data_names)

	evaluator = DimOriginCltEvaluator()
	evaluator.gen_summary_csv([
			'phenograph',
			'kmeans (random; 1)',
			'kmeans (random; 10)',
			'kmeans (k-means++; 1)',
			'kmeans (k-means++; 10)',
			'rp kmeans (default; 1)'
		],
		data_names=data_names,
		repeat_times=repeat_times
	)


