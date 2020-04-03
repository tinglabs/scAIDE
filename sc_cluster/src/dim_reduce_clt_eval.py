"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import os
# To precisely count the clustering CPU time, activate the following 3 lines when running clustering. And remember to
# deactivate them when running dimension reduction. It's amazing that limiting the threads number using the
# following 3 lines can accelerate kmeans(init with kmeans++) and rp-kmeans dramatically on linux OS (24 cores).
# It's probably because 'numpy' takes too much CPU time in creating and destroying new threads and transfering
# the data between threads when performing matrix calculation on a multi-core machine. The acceleration will be
# more significant when matrix.shape[1] is small.
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from tqdm import tqdm
from copy import deepcopy
import pandas as pd
import scipy
import scipy.sparse as sp
import random
import itertools
import json
from time import time, process_time
import numpy as np
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

from preprocess import get_process_data, get_process_labels, get_process_features, get_process_data_info
from data_explain import AryDataExplainer
from aide.aide_ import AIDE, AIDEConfig
from aide.deep_mds import DeepMDS, DeepMDSConfig
from aide.ae import AE, AEConfig
from rp_kmeans import RPKMeans, select_k_with_bic
from constant import RESULT_PATH, DATA_PATH, MODEL_PATH, TEMP_PATH, EMBEDDING_PATH
from aide.constant import DATA_MAT, DATA_TFRECORD
from utils_draw import simple_dot_plot, simple_dist_plot
from utils_ import unzip_dict


class DimReduceCltEvaluator(object):
	def __init__(self):
		self.SAVE_FOLDER = RESULT_PATH + os.sep + 'dim_reduce_eval'

		self.DRAW_FOLDER = self.SAVE_FOLDER + os.sep + 'draw'
		os.makedirs(self.DRAW_FOLDER, exist_ok=True)

		self.METRIC_FOLDER = self.SAVE_FOLDER + os.sep + 'metric'
		os.makedirs(self.METRIC_FOLDER, exist_ok=True)

		self.SAMPLE_RANKS_FOLDER = self.SAVE_FOLDER + os.sep + 'sample_ranks'
		os.makedirs(self.SAMPLE_RANKS_FOLDER, exist_ok=True)

		self.EMBED_INFO_FOLDER = self.SAVE_FOLDER + os.sep + 'embedding_info'
		os.makedirs(self.EMBED_INFO_FOLDER, exist_ok=True)

		self.K_SELECTION_FOLDER = self.SAVE_FOLDER + os.sep + 'k_selection'
		os.makedirs(self.K_SELECTION_FOLDER, exist_ok=True)

		os.makedirs(MODEL_PATH, exist_ok=True)
		os.makedirs(EMBEDDING_PATH, exist_ok=True)


	def get_embedding_path(self, encoder_name, embed_repeat_id):
		return EMBEDDING_PATH + os.sep + '{}-{}-embedding.npy'.format(encoder_name, embed_repeat_id)


	def get_embedding_2d_npy(self, encoder_name, embed_repeat_id, method, sample_num):
		return os.path.join(self.DRAW_FOLDER, encoder_name, f'{method}-{embed_repeat_id}-{sample_num}.npy')


	def get_encoder_save_folder(self, encoder_name):
		return MODEL_PATH + os.sep + encoder_name


	def get_fig_save_folder(self, encoder_name, mark=''):
		return os.path.join(self.DRAW_FOLDER, encoder_name, mark) if mark else os.path.join(self.DRAW_FOLDER, encoder_name)


	def get_raw_draw_folder(self, data_name):
		return self.DRAW_FOLDER + os.sep + '{}-{}'.format('RAW', data_name)


	def get_rp_kmeans_clt_name(self, max_point=2000, w=None, proj_num=5, n_init=1, bkt_improve=None,
			radius_divide=None, bkt_size_keepr=1.0, center_dist_keepr=1.0, **kwargs):
		return 'RPKMeans_mp{}_w{}_pj{}_ninit{}_bkt{}_{}_{}_{}_{}'.format(
				max_point, w, proj_num, n_init, bkt_improve, radius_divide, bkt_size_keepr, center_dist_keepr, kwargs)


	def get_draw_sample_ranks(self, data_name, sample_num=1000, data_size=None):
		path = self.SAMPLE_RANKS_FOLDER + os.sep + '{}-{}.npy'.format(data_name, sample_num)
		if os.path.exists(path):
			return np.load(path)
		data_size = data_size or get_process_data_info(data_name)['CELL_NUM']
		sample_ranks = np.random.choice(data_size, sample_num, replace=False)
		np.save(path, sample_ranks)
		return sample_ranks


	# get embedding ================================================================
	def get_pca_embedding(self, X, dim):
		"""
		Args:
			X (np.ndarray or sp.csr_matrix)
		Returns:
			np.ndarray: (sample_num, embed_size)
		"""
		if sp.issparse(X):
			X = X.A
		return PCA(n_components=dim, random_state=random.randint(1, 10000)).fit_transform(X)


	def get_mds_embedding(self, X, dim):
		X = X.astype(np.float64).A if sp.issparse(X) else X.astype(np.float64)
		return MDS(n_components=dim).fit_transform(X)


	def get_dime_embedding(self, X, config, encoder_name, save_folder):
		encoder = AIDE(encoder_name, save_folder)
		embedding = encoder.fit_transform(X, config=config, from_last=False)
		return embedding


	def get_deep_mds_embedding(self, X, config, encoder_name, save_folder):
		encoder = DeepMDS(encoder_name, save_folder)
		embedding = encoder.fit_transform(X, config=config, from_last=False)
		return embedding


	def get_ae_embedding(self, X, config, encoder_name, save_folder):
		encoder = AE(encoder_name, save_folder)
		embedding = encoder.fit_transform(X, config=config, from_last=False)
		return embedding


	# embedding eval ================================================================
	def draw_embedding_l2_norm(self, figpath, X, bins=100, sample_ranks=None):
		sample_ranks = np.arange(X.shape[0]) if sample_ranks is None else sample_ranks
		embed = X[sample_ranks]
		x = np.sqrt(np.power(embed, 2).sum(1))
		simple_dist_plot(figpath, x, bins, 'l2_norm', 'Embedding L2Norm Distribution')


	def draw_embedding_ele_range(self, figpath, X, bins=100, sample_ranks=None):
		sample_ranks = np.arange(X.shape[0]) if sample_ranks is None else sample_ranks
		embed = X[sample_ranks]
		simple_dist_plot(figpath, embed.flatten(), bins, 'Element', 'Embedding Element Distribution')


	def get_dim_reduction(self, data_name, encoder_name, embed_repeat_id, sample_num=1000, method='tsne'):
		x_2d_save_npy = self.get_embedding_2d_npy(encoder_name, embed_repeat_id, method, sample_num)
		if os.path.exists(x_2d_save_npy):
			return np.load(x_2d_save_npy)
		embedding = np.load(self.get_embedding_path(encoder_name, embed_repeat_id))
		sample_ranks = self.get_draw_sample_ranks(data_name, data_size=embedding.shape[0], sample_num=sample_num)
		X = np.load(self.get_embedding_path(encoder_name, embed_repeat_id))[sample_ranks]
		if method == 'tsne':
			x_2d = TSNE().fit_transform(X) if X.shape[1] > 2 else X
		elif method == 'pca':
			x_2d = PCA(n_components=2).fit_transform(X) if X.shape[1] > 2 else X
		else:
			raise RuntimeError('Unknown dimension reduction: {}'.format(method))
		np.save(x_2d_save_npy, x_2d)
		return x_2d


	def draw_tsne_reduction(self, figpath, title, X, y=None, sample_ranks=None):
		sample_ranks = np.arange(X.shape[0]) if sample_ranks is None else sample_ranks
		X = X[sample_ranks]
		X = X.toarray() if sp.issparse(X) else X
		y = None if y is None else y[sample_ranks]
		tsne_x = TSNE().fit_transform(X) if X.shape[1] > 2 else X
		simple_dot_plot(
			figpath, tsne_x[:, 0], tsne_x[:, 1],
			p_types=y, p_type_label=None if y is None else 'label',
			title=title, figsize=(20, 20))


	def draw_pca_reduction(self, figpath, title, X, y=None, sample_ranks=None):
		sample_ranks = np.arange(X.shape[0]) if sample_ranks is None else sample_ranks
		X = X[sample_ranks]
		X = X.toarray() if sp.issparse(X) else X
		y = None if y is None else y[sample_ranks]
		pca_x = PCA(n_components=2).fit_transform(X) if X.shape[1] > 2 else X
		simple_dot_plot(
			figpath, pca_x[:, 0], pca_x[:, 1],
			p_types=y, p_type_label=None if y is None else 'label',
			title=title, figsize=(20, 20))


	# clt eval ================================================================
	def get_predict_performance(self, args):
		"""
		Returns:
			dict: {
				'ARI': float
				'NMI': float
				'CLUSTER_TIME'
			}
		"""
		scipy.random.seed() # Important
		X, y_true, clt_initializer, clt_kwargs, eval_kwargs = args

		p_time, t_time = process_time(), time()
		clt = clt_initializer(**clt_kwargs)
		y_pred = clt.fit_predict(X)
		p_time_spend, t_time_spend = process_time() - p_time, time() - t_time

		d = {
			'CLUSTER_REAL_TIME': t_time_spend,
			'CLUSTER_CPU_TIME': p_time_spend,
			'ARI': adjusted_rand_score(y_true, y_pred),
			'NMI': normalized_mutual_info_score(y_true, y_pred),
		}
		print(d)

		if eval_kwargs:
			print('drawing...')
			figfolder = eval_kwargs['FIG_FOLDER']
			os.makedirs(figfolder, exist_ok=True)
			sample_ranks = np.random.choice(X.shape[0], 1000, replace=False)
			X, y_true, y_pred = X[sample_ranks], y_true[sample_ranks], y_pred[sample_ranks]
			X = X.toarray() if sp.issparse(X) else X
			tsne_x = TSNE().fit_transform(X) if X.shape[1] > 2 else X
			self.draw_tsne_reduction(os.path.join(figfolder, 'tsne_y_pred'), 'y_pred (ARI: {:.4f}; NMI: {:.4f})'.format(d['ARI'], d['NMI']), tsne_x, y_pred)
			self.draw_tsne_reduction(os.path.join(figfolder, 'tsne_y_true'), 'y_true', tsne_x, y_true)
		return d


	def combine_metric_dicts(self, dlist):
		"""
		Args:
			dlist (list): [{metric_name: score}, ...]
		Returns:
			dict: {metric_name: score_mean (score_std)}
		"""
		d = {}
		for k in dlist[0]:
			score_list = [metric_dict[k] for metric_dict in dlist]
			ave = np.mean(score_list)
			std = np.std(score_list)
			d[k] = '{:.3} ({:.3})'.format(ave, std)
		return d


	def combine_result_base(self, results):
		final_result = []
		clt_names = results[0].keys()
		for clt_name in clt_names:
			metric_dicts = []
			for result in results:
				metric_dicts.extend(result[clt_name])
			d = self.combine_metric_dicts(metric_dicts)
			d['CLT_NAME'] = clt_name
			final_result.append(d)
		return final_result


	def combine_result(self, encoder_name, clt_mark, embed_repeat=5):
		"""
		Args:
			clt_mark (str): e.g. 'rp-kmeans' | 'kmeans (k-means++)' | 'kmeans (random)' | 'rp-kmeans (default)' | rp-kmeans (best)'
		"""
		save_folder = os.path.join(self.METRIC_FOLDER, encoder_name, clt_mark)
		results = [json.load(open(save_folder + os.sep + 'embedding-{}.json'.format(i))) for i in range(embed_repeat)]

		final_result = self.combine_result_base(results)
		for clt_result in final_result:
			print('{}: {}'.format(clt_result['CLT_NAME'], clt_result))

		pd.DataFrame(final_result).to_csv(
			save_folder + '/final_results.csv', index=False,
			columns=['CLT_NAME', 'CLUSTER_CPU_TIME', 'CLUSTER_REAL_TIME', 'ARI', 'NMI'])


	def combine_time_result(self, encoder_name, embed_repeat=2):
		save_folder = os.path.join(self.METRIC_FOLDER, encoder_name, 'dim_reduction_time')
		dlist = [json.load(open(os.path.join(save_folder, 'embedding-{}.json'.format(i)))) for i in range(embed_repeat)]
		final_result = self.combine_metric_dicts(dlist)
		pd.DataFrame([final_result]).to_csv(
			save_folder + '/final_results.csv', index=False,
			columns=['EMBEDDING_CPU_TIME', 'EMBEDDING_REAL_TIME'])


	def prepare_x(self, data_name, data_type):
		X = get_process_features(data_name, data_type)
		info_dict = get_process_data_info(data_name, data_type)
		if data_type == DATA_MAT:
			if sp.issparse(X) and info_dict['FEATURE_NON_ZERO_RATIO'] > 0.2 and info_dict['CELL_NUM'] < 5000:
				X = X.toarray()
			return X
		elif data_type == DATA_TFRECORD:
			return X, {'n_samples': info_dict['CELL_NUM'], 'n_features': info_dict['GENE_NUM'], 'issparse': True}
		else:
			raise RuntimeError('Unknown data type: {}'.format(data_type))


	# running ================================================================
	def run_all_embedding(self, data_name, encoder_name, embed_method, embed_repeat=5, data_type=DATA_MAT, config=None, save_embedding=True, **kwargs):
		"""
		Args:
			embed_method (str): 'PCA' | 'MDS' | 'AIDE' | 'DeepMDS'
		Returns:
			list : [{
				'EMBEDDING_CPU_TIME': float,
				'EMBEDDING_REAL_TIME': float,
			}, ...]
		"""
		print('Run embedding: {}'.format(encoder_name))
		X = self.prepare_x(data_name, data_type)

		for embed_repeat_id in range(embed_repeat):
			cpu_t, real_t = process_time(), time()
			if embed_method == 'PCA':
				assert not isinstance(X, str)
				embedding = self.get_pca_embedding(X, kwargs.get('dim', 256))
			elif embed_method == 'MDS':
				assert not isinstance(X, str)
				embedding = self.get_mds_embedding(X, kwargs.get('dim', 256))
			elif embed_method == 'AIDE':
				model_save_folder = self.get_encoder_save_folder(encoder_name)
				c = deepcopy(config) if config else AIDEConfig()
				embedding = self.get_dime_embedding(X, c, encoder_name, model_save_folder)
			elif embed_method == 'DeepMDS':
				model_save_folder = self.get_encoder_save_folder(encoder_name)
				c = deepcopy(config) if config else DeepMDSConfig()
				embedding = self.get_deep_mds_embedding(X, c, encoder_name, model_save_folder)
			elif embed_method == 'AE':
				model_save_folder = self.get_encoder_save_folder(encoder_name)
				c = deepcopy(config) if config else AEConfig()
				embedding = self.get_ae_embedding(X, c, encoder_name, model_save_folder)
			else:
				raise RuntimeError('Unknown embed_method: {}. Should be one of ["PCA", "MDS", "AIDE", "DeepMDS", "AE"]'.format(embed_method))
			embedding_cpu_time, embedding_real_time = process_time() - cpu_t, time() - real_t
			print('EMBEDDING_CPU_TIME:', embedding_cpu_time, '; EMBEDDING_REAL_TIME:', embedding_real_time)
			time_dict = {'EMBEDDING_CPU_TIME': embedding_cpu_time, 'EMBEDDING_REAL_TIME': embedding_real_time}
			save_json = os.path.join(self.METRIC_FOLDER, encoder_name, 'dim_reduction_time', 'embedding-{}.json'.format(embed_repeat_id))
			os.makedirs(os.path.dirname(save_json), exist_ok=True)
			json.dump(time_dict, open(save_json, 'w'), indent=2)

			if save_embedding:
				embedding_save_path = self.get_embedding_path(encoder_name, embed_repeat_id)
				np.save(embedding_save_path, embedding)


	def eval_raw(self, data_name):
		fig_save_folder = self.get_raw_draw_folder(data_name)
		os.makedirs(fig_save_folder, exist_ok=True)
		X, y_true = get_process_data(data_name, data_type=DATA_MAT)
		sample_ranks = self.get_draw_sample_ranks(data_name, data_size=X.shape[0])

		self.draw_tsne_reduction(
			fig_save_folder + os.sep + 'tsne_raw_features.png', 'tsne_raw_features',
			X, y_true, sample_ranks=sample_ranks)
		self.draw_pca_reduction(
			fig_save_folder + os.sep + 'pca_raw_features.png', 'pca_raw_features',
			X, y_true, sample_ranks=sample_ranks)


	def eval_embedding(self, data_name, encoder_name, embed_repeat=5):
		fig_save_folder = self.get_fig_save_folder(encoder_name)
		os.makedirs(fig_save_folder, exist_ok=True)

		for embed_repeat_id in range(embed_repeat):
			embedding = np.load(self.get_embedding_path(encoder_name, embed_repeat_id))
			y_true = get_process_labels(data_name)
			sample_ranks = self.get_draw_sample_ranks(data_name, data_size=embedding.shape[0])

			embed_info = AryDataExplainer(embedding, y_true).explain()
			json.dump(embed_info, open(self.EMBED_INFO_FOLDER + os.sep + f'{encoder_name}-{embed_repeat_id}.json', 'w'), indent=2)

			self.draw_embedding_l2_norm(
				fig_save_folder + os.sep + 'l2_norm_embedding-{}.png'.format(embed_repeat_id),
				embedding, sample_ranks=sample_ranks)
			self.draw_embedding_ele_range(
				fig_save_folder + os.sep + 'ele_range_embedding-{}.png'.format(embed_repeat_id),
				embedding, sample_ranks=sample_ranks)
			self.draw_tsne_reduction(
				fig_save_folder + os.sep + 'tsne_embedding-{}.png'.format(embed_repeat_id),
				'tsne_embedding', embedding, y_true, sample_ranks=sample_ranks)
			self.draw_pca_reduction(
				fig_save_folder + os.sep + 'pca_embedding-{}.png'.format(embed_repeat_id),
				'pca_embedding', embedding, y_true, sample_ranks=sample_ranks)


	def run_clt_base(self, clt_name_to_clt_args, embedding, y_true, clt_repeat, clt_name_to_eval_kwargs=None):
		"""
		Args:
			clt_name_to_clt_args (dict): {clt_name: (clt_initializer, clt_kwargs)}
		Returns:
			dict: {
				clt_name: [metric_dict, ...]
			}
		"""
		ret_dict = {}
		for clt_name, (clt_initializer, clt_kwargs) in clt_name_to_clt_args.items():
			eval_kwargs = clt_name_to_eval_kwargs.get(clt_name, {})
			# multiprocess
			with Pool(10) as pool:
				dlist = pool.map(self.get_predict_performance,
					[(embedding, y_true, clt_initializer, clt_kwargs, eval_kwargs) for i in range(clt_repeat)])
			# # single process
			# dlist = [self.get_predict_performance(
			# 	(embedding, y_true, clt_initializer, clt_kwargs, eval_kwargs)) for i in range(clt_repeat)]

			ret_dict[clt_name] = dlist
			print('## {} ##'.format(clt_name), self.combine_metric_dicts(dlist))
		return ret_dict


	def run_kmeans_clt(self, data_name, encoder_name, embed_repeat_id, clt_repeat=10, draw=False):
		embed_path = self.get_embedding_path(encoder_name, embed_repeat_id)
		print('loading ', embed_path)
		embedding = np.load(embed_path)
		y_true = get_process_labels(data_name)
		n_clusters = len(np.unique(y_true))
		print('data_size={}, n_clusters={}'.format(embedding.shape, n_clusters))

		for init_type in ['k-means++', 'random']:
			clt_name_to_clt_args, clt_name_to_eval_kwargs = {}, {}
			n_init_list = [1, 3, 5, 8, 10, 15, 20, 25, 30, 50, 70, 100]
			for n_init in n_init_list:
				clt_name = 'kmeans ({}; {})'.format(init_type, n_init)
				clt_initializer, clt_kwargs = KMeans, {'n_clusters': n_clusters, 'init': init_type, 'n_init': n_init}
				clt_name_to_clt_args[clt_name] = (clt_initializer, clt_kwargs)
				clt_name_to_eval_kwargs[clt_name] = {'FIG_FOLDER':os.path.join(self.get_fig_save_folder(encoder_name, f'{clt_name}-{embed_repeat_id}'))}

			ret_dict = self.run_clt_base(clt_name_to_clt_args, embedding, y_true, clt_repeat, clt_name_to_eval_kwargs if draw else {})

			save_json = os.path.join(self.METRIC_FOLDER, encoder_name,
				'kmeans ({})'.format(init_type), 'embedding-{}.json'.format(embed_repeat_id))
			os.makedirs(os.path.dirname(save_json), exist_ok=True)
			json.dump(ret_dict, open(save_json, 'w'), indent=2)


	def run_rp_kmeans_clt(self, data_name, encoder_name, embed_repeat_id, clt_repeat=10, draw=False):
		embedding = np.load(self.get_embedding_path(encoder_name, embed_repeat_id))
		y_true = get_process_labels(data_name)
		n_clusters = len(np.unique(y_true))
		print('data_size={}, n_clusters={}'.format(embedding.shape, n_clusters))

		clt_name_to_clt_args, clt_name_to_eval_kwargs = {}, {}
		for max_point, w, proj_num, n_init, bkt_improve, radius_divide, bkt_size_keepr, center_dist_keepr in itertools.product(
				[2000], [None], [5], [1], [None], [None], [1.0], [1.0]):
			clt_name = self.get_rp_kmeans_clt_name(max_point, w, proj_num, n_init,
				bkt_improve, radius_divide, bkt_size_keepr, center_dist_keepr)
			clt_initializer = RPKMeans
			clt_kwargs = {'n_clusters':n_clusters, 'max_point':max_point, 'w': w, 'proj_num': proj_num, 'n_init': n_init,
				'bkt_improve': bkt_improve, 'radius_divide': radius_divide, 'bkt_size_keepr': bkt_size_keepr,
				'center_dist_keepr': center_dist_keepr}
			clt_name_to_clt_args[clt_name] = (clt_initializer, clt_kwargs)
			clt_name_to_eval_kwargs[clt_name] = {'FIG_FOLDER':os.path.join(self.get_fig_save_folder(encoder_name, f'{clt_name}-{embed_repeat_id}'))}

		ret_dict = self.run_clt_base(clt_name_to_clt_args, embedding, y_true, clt_repeat, clt_name_to_eval_kwargs if draw else {})

		save_json = os.path.join(self.METRIC_FOLDER, encoder_name, 'rp-kmeans', 'embedding-{}.json'.format(embed_repeat_id))
		os.makedirs(os.path.dirname(save_json), exist_ok=True)
		json.dump(ret_dict, open(save_json, 'w'), indent=2)


	def run_all_embedding_clt(self, run_clt_func, data_name, encoder_name, embed_repeat=5, clt_repeat=10, draw=False):
		for embed_repeat_id in range(embed_repeat):
			print('Embedding Repeat: {} ====================================================='.format(embed_repeat_id))
			run_clt_func(data_name, encoder_name, embed_repeat_id, clt_repeat, draw)


	def run_rp_kmeans_with_dict(self, data_name, encoder_name, clt_kwargs_list=None,
			clt_mark='rp-kmeans', embed_repeat=5, clt_repeat=10, draw=False):
		y_true = get_process_labels(data_name)
		n_clusters = len(np.unique(y_true))

		clt_initializer = RPKMeans
		clt_kwargs_list = clt_kwargs_list or [{}]

		results = []
		save_folder = os.path.join(self.METRIC_FOLDER, encoder_name, clt_mark); os.makedirs(save_folder, exist_ok=True)
		for embed_repeat_id in range(embed_repeat):
			embed_path = self.get_embedding_path(encoder_name, embed_repeat_id)
			print('loading ', embed_path)
			embedding = np.load(embed_path)
			print('Embedding Repeat: {} ====================================================='.format(embed_repeat_id))
			print('data_size={}, n_clusters={}'.format(embedding.shape, n_clusters))

			clt_name_to_clt_args, clt_name_to_eval_kwargs = {}, {}
			for clt_kwargs in clt_kwargs_list:
				if 'n_clusters' not in clt_kwargs:
					clt_kwargs['n_clusters'] = n_clusters
				clt_name = self.get_rp_kmeans_clt_name(**clt_kwargs)
				clt_name_to_clt_args[clt_name] = (clt_initializer, clt_kwargs)
				clt_name_to_eval_kwargs[clt_name] = {'FIG_FOLDER':os.path.join(self.get_fig_save_folder(encoder_name, f'{clt_name}-{embed_repeat_id}'))}

			result = self.run_clt_base(clt_name_to_clt_args, embedding, y_true, clt_repeat, clt_name_to_eval_kwargs if draw else {})
			results.append(result)
			save_json = os.path.join(save_folder, 'embedding-{}.json'.format(embed_repeat_id))
			json.dump(result, open(save_json, 'w'), indent=2)


	def run_k_metric_wrapper(self, args):
		scipy.random.seed()
		embedding, y_true, clt_initializer, clt_kwargs = args
		clt = clt_initializer(**clt_kwargs)
		y_pred = clt.fit_predict(embedding)
		return adjusted_rand_score(y_true, y_pred), normalized_mutual_info_score(y_true, y_pred)


	def run_k_metric(self, data_name, encoder_name, embed_repeat_id, clt_name, clt_initializer, clt_kwargs=None, clt_repeat=5, cpu_use=20):
		print(f'run_k_metric: {data_name}, {encoder_name}, {clt_name}')
		save_folder = os.path.join(self.K_SELECTION_FOLDER, encoder_name, f'embedding-{embed_repeat_id}')
		os.makedirs(save_folder, exist_ok=True)
		y_true = get_process_labels(data_name)
		n_clusters_true = len(np.unique(y_true))

		embed_path = self.get_embedding_path(encoder_name, embed_repeat_id)
		embedding = np.load(embed_path)
		kmax = min(max(n_clusters_true * 3, 10), 40)
		k_range = list(range(2, kmax + 1))
		clt_kwargs = clt_kwargs or {}
		ari_lists, nmi_lists = [], []
		for clt_repeat_id in range(clt_repeat):
			ari_list, nmi_list = [], []
			args = [(embedding, y_true, clt_initializer, {**clt_kwargs, **{'n_clusters': k}}) for k in k_range]
			with Pool(cpu_use) as pool:
				for ari, nmi in tqdm(pool.imap(self.run_k_metric_wrapper, args), total=len(args), leave=False):
					ari_list.append(ari); nmi_list.append(nmi)
			ari_lists.append(ari_list); nmi_lists.append(nmi_list)
		json.dump({'K_RANGE': k_range, 'ARI': ari_lists}, open(os.path.join(save_folder, f'ARI-{clt_name}.json'), 'w'), indent=2)
		json.dump({'K_RANGE': k_range, 'NMI': nmi_lists}, open(os.path.join(save_folder, f'NMI-{clt_name}.json'), 'w'), indent=2)


	def run_k_selection(self, data_name, encoder_name, embed_repeat_id=0, point_reducer_kwargs=None, cpu_use=20):
		print(f'run_k_selection: {data_name}, {encoder_name}')
		save_folder = os.path.join(self.K_SELECTION_FOLDER, encoder_name, f'embedding-{embed_repeat_id}')
		os.makedirs(save_folder, exist_ok=True)
		y_true = get_process_labels(data_name)
		n_clusters_true = len(np.unique(y_true))

		embed_path = self.get_embedding_path(encoder_name, embed_repeat_id)
		embedding = np.load(embed_path)
		kmax = min(max(n_clusters_true * 3, 10), 40)
		n_clusters_pred, bic_lists, k_range = select_k_with_bic(
			embedding, kmax=kmax, point_reducer_kwargs=point_reducer_kwargs, n_jobs=cpu_use)

		json.dump(
			{'K_RANGE': k_range, 'BIC_LISTS': bic_lists, 'N_CLUSTERS_TRUE': n_clusters_true, 'N_CLUSTERS_PRED': n_clusters_pred},
			open(os.path.join(save_folder, 'BIC.json'), 'w'), indent=2)


if __name__ == '__main__':
	from reader import get_all_data_names, get_all_normal_data_names, get_all_imb_data_names, is_extreme_large
	from reader import get_all_sample_data_names, get_all_sim_dropout_data_names

	def get_encoder_name(embed_method, data_name, dim=256, mark=''):
		s = '{}-{}-{}'.format(embed_method, dim, data_name)
		if mark:
			s += f'-{mark}'
		return s

	evaluator = DimReduceCltEvaluator()
	embed_repeat = 5
	clt_repeat = 10

	# Evaluate all raw data =========================================================
	data_names = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']
	for data_name in data_names:
		print(data_name)
		evaluator.eval_raw(data_name)

	# monitor memory:  mprof run --python -C python dim_reduce_clt_eval.py && mprof plot -o mem.png
	print('PCA/MDS Embedding =========================================================')
	data_names = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']
	embed_methods = ['PCA']
	reduced_dim = 256
	for embed_method, data_name in itertools.product(embed_methods, data_names):
		encoder_name = get_encoder_name(embed_method, data_name, reduced_dim)
		evaluator.run_all_embedding(
			data_name, encoder_name, embed_method, embed_repeat=embed_repeat, data_type=DATA_MAT, dim=reduced_dim)
		evaluator.eval_embedding(data_name, encoder_name, embed_repeat=embed_repeat)


	print('PCA/MDS + KMeans ------------------------')
	data_names = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']
	embed_methods = ['PCA']
	reduced_dim = 256
	for embed_method, data_name in itertools.product(embed_methods, data_names):
		encoder_name = get_encoder_name(embed_method, data_name, reduced_dim)
		evaluator.run_all_embedding_clt(
			evaluator.run_kmeans_clt, data_name, encoder_name,
			embed_repeat=embed_repeat, clt_repeat=clt_repeat)
		evaluator.combine_result(encoder_name, 'kmeans ({})'.format('random'), embed_repeat=embed_repeat)
		evaluator.combine_result(encoder_name, 'kmeans ({})'.format('k-means++'), embed_repeat=embed_repeat)


	print('PCA + RPKMeans (best) ------------------------')
	PCA_RP_KMEANS_CONFIG_DICT = {
		'sc_brain':{
			'proj_num': 7,
		},
		'PBMC_68k':{
			'max_point': 4000,
			'w': 5.5,
			'proj_num': 6,
		},
		'Shekhar_mouse_retina':{
			'bkt_improve': 'radius',
			'radius_divide': 13.5,
			'w': 14.0,
		},
		'10X_PBMC':{
			'bkt_improve': 'radius',
			'radius_divide': 13.0,
		},
		'mouse_bladder_cell':{
			'max_point': 500,
			'bkt_improve': 'min_bkt_size',
			'bkt_size_keepr': 0.8,
		},
		'mouse_ES_cell':{
			'bkt_improve': 'radius',
			'radius_divide':15.0,
		},
		'worm_neuron_cell':{
			'bkt_improve': 'radius',
			'radius_divide': 7.5,
		}
	}
	data_names = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']
	reduced_dim = 256
	for data_name in data_names:
		encoder_name = get_encoder_name('PCA', data_name, dim=reduced_dim)
		clt_mark = 'rp-kmeans (best)'
		d = PCA_RP_KMEANS_CONFIG_DICT[data_name]
		d['verbose'] = 0; d['n_init'] = [1, 2, 3, 4, 5, 8, 10, 15, 20]
		evaluator.run_rp_kmeans_with_dict(data_name, encoder_name,
			clt_kwargs_list=unzip_dict(d),
			clt_mark=clt_mark, embed_repeat=embed_repeat, clt_repeat=clt_repeat)
		evaluator.combine_result(encoder_name, clt_mark, embed_repeat=embed_repeat)


	print('AIDE (best) Embedding =========================================================')
	AIDE_CONFIG_DICT = {
		'sc_brain':{
			'alpha': 40.0,
		},
		'PBMC_68k':{
			'alpha': 1.0,
			'pretrain_step_num': 2000,
		},
		'Shekhar_mouse_retina':{
			'alpha': 20.0,
			'pretrain_step_num':2000,
		},
		'10X_PBMC':{
			'alpha':1.0,
			'pretrain_step_num':2000,
			'ae_drop_out_rate':0.2,
		},
		'mouse_bladder_cell':{
			'alpha': 40.0,
			'pretrain_step_num': 2000,
			'early_stop_patience': None,
		},
		'mouse_ES_cell':{
			'alpha': 15.0,
		},
		'worm_neuron_cell':{
			'alpha': 20.0,
			'ae_drop_out_rate': 0.5,
		}
	}
	data_names = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']
	for data_name in data_names:
		encoder_name = get_encoder_name('AIDE', data_name, mark='Best')
		config = AIDEConfig(assign_dict=AIDE_CONFIG_DICT[data_name])
		evaluator.run_all_embedding(
			data_name, encoder_name, 'AIDE', config=config,
			embed_repeat=embed_repeat, data_type=DATA_MAT)
		evaluator.eval_embedding(data_name, encoder_name, embed_repeat=embed_repeat)


	print('AIDE (best) + KMeans ------------------------')
	data_names = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']
	for data_name in data_names:
		encoder_name = get_encoder_name('AIDE', data_name, mark='Best')
		evaluator.run_all_embedding_clt(
			evaluator.run_kmeans_clt, data_name, encoder_name,
			embed_repeat=embed_repeat, clt_repeat=clt_repeat)
		evaluator.combine_result(encoder_name, 'kmeans ({})'.format('random'), embed_repeat=embed_repeat)
		evaluator.combine_result(encoder_name, 'kmeans ({})'.format('k-means++'), embed_repeat=embed_repeat)


	print('AIDE (best) + RPKMeans (best) ------------------------')
	AIDE_RP_KMEANS_CONFIG_DICT = {
		'sc_brain':{},
		'PBMC_68k':{
			'max_point': 6000,
		},
		'Shekhar_mouse_retina':{
			'proj_num': 15,
		},
		'10X_PBMC':{
			'bkt_improve':'min_bkt_size',
			'bkt_size_keepr': 0.5,
		},
		'mouse_bladder_cell':{
			'w': 0.5,
			'proj_num': 10,
		},
		'mouse_ES_cell':{},
		'worm_neuron_cell':{
			'bkt_improve':'min_bkt_size',
			'bkt_size_keepr':0.5,
		}
	}
	data_names = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']
	for data_name in data_names:
		encoder_name = get_encoder_name('AIDE', data_name, mark='Best')
		clt_mark = 'rp-kmeans (best)'
		d = AIDE_RP_KMEANS_CONFIG_DICT[data_name]
		d['verbose'] = 0; d['n_init'] = [1, 2, 3, 4, 5, 8, 10, 15, 20]
		evaluator.run_rp_kmeans_with_dict(data_name, encoder_name,
			clt_kwargs_list=unzip_dict(d),
			clt_mark=clt_mark, embed_repeat=embed_repeat, clt_repeat=clt_repeat)
		evaluator.combine_result(encoder_name, clt_mark, embed_repeat=embed_repeat)


	print('AIDE (best) Sparsity Simulation Embedding =========================================================')
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	AIDE_SIM_SPARSITY_CONFIG_DICT = {
		'early_stop_patience': None,
		'max_step_num': 40000,
	}
	data_names = get_all_sim_dropout_data_names([60, 70, 75, 80, 85, 90, 93])
	for data_name in data_names:
		encoder_name = get_encoder_name('AIDE', data_name, mark='Default-40000')
		config = AIDEConfig(assign_dict=AIDE_SIM_SPARSITY_CONFIG_DICT)
		evaluator.run_all_embedding(
			data_name, encoder_name, 'AIDE', config=config,
			embed_repeat=embed_repeat, data_type=DATA_MAT)
		evaluator.eval_embedding(data_name, encoder_name, embed_repeat=embed_repeat)


	print('AIDE (default) Embedding =========================================================')
	data_names = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']
	for data_name in data_names:
		encoder_name = get_encoder_name('AIDE', data_name, mark='Default')
		config = AIDEConfig()
		evaluator.run_all_embedding(
			data_name, encoder_name, 'AIDE', embed_repeat=embed_repeat, data_type=DATA_MAT, config=config)
		evaluator.eval_embedding(data_name, encoder_name, embed_repeat=embed_repeat)


	print('AIDE (default) + KMeans ------------------------')
	data_names = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']
	for data_name in data_names:
		encoder_name = get_encoder_name('AIDE', data_name, mark='Default')
		evaluator.run_all_embedding_clt(
			evaluator.run_kmeans_clt, data_name, encoder_name,
			embed_repeat=embed_repeat, clt_repeat=clt_repeat, draw=False)
		evaluator.combine_result(encoder_name, 'kmeans ({})'.format('random'), embed_repeat=embed_repeat)
		evaluator.combine_result(encoder_name, 'kmeans ({})'.format('k-means++'), embed_repeat=embed_repeat)


	print('AIDE (default) + RPKMeans (best) ------------------------')
	AIDE_DEFAULT_RP_KMEANS_BEST_CONFIG_DICT = {
		'PBMC_68k':{
			'w': 2.0,
			'bkt_improve':'min_bkt_size',
			'bkt_size_keepr': 0.9,
		},
		'10X_PBMC':{
			'w': 0.5,
		},
	}
	for data_name in AIDE_DEFAULT_RP_KMEANS_BEST_CONFIG_DICT:
		encoder_name = get_encoder_name('AIDE', data_name, mark='Default')
		clt_mark = 'rp-kmeans (best)'
		d = AIDE_DEFAULT_RP_KMEANS_BEST_CONFIG_DICT.get(data_name, {})
		d['verbose'] = 0; d['n_init'] = [1, 2, 3, 4, 5, 8, 10, 15, 20]
		evaluator.run_rp_kmeans_with_dict(data_name, encoder_name,
			clt_kwargs_list=unzip_dict(d),
			clt_mark=clt_mark, embed_repeat=embed_repeat, clt_repeat=clt_repeat)
		evaluator.combine_result(encoder_name, clt_mark, embed_repeat=embed_repeat)


	DEFAULT_RP_KMEANS_CONFIG_DICT = {
		'verbose': 0,
		'n_init':[1, 10],
		# 'n_init': [1, 3, 5, 8, 10, 15, 20, 25, 30, 50, 70, 100],
	}
	print('RPKMeans (default) ------------------------')
	embed_configs = [('PCA', ''), ('AIDE', 'Best')]
	data_names = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']
	for embed_method, mark in embed_configs:
		for data_name in data_names:
			encoder_name = get_encoder_name(embed_method, data_name, mark=mark)
			clt_mark = 'rp-kmeans (default)'
			evaluator.run_rp_kmeans_with_dict(data_name, encoder_name,
				clt_kwargs_list=unzip_dict(DEFAULT_RP_KMEANS_CONFIG_DICT),
				clt_mark=clt_mark, embed_repeat=embed_repeat, clt_repeat=clt_repeat)
			evaluator.combine_result(encoder_name, clt_mark, embed_repeat=embed_repeat)


	print('DeepMDS (default) Embedding =========================================================')
	data_names = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']
	for data_name in data_names:
		encoder_name = get_encoder_name('DeepMDS', data_name, mark='Default')
		config = DeepMDSConfig()
		result, ret_dict = evaluator.run_all_embedding(
			data_name, encoder_name, 'DeepMDS', embed_repeat=embed_repeat, data_type=DATA_MAT, config=config)
		evaluator.eval_embedding(data_name, encoder_name, embed_repeat=embed_repeat)


	print('AE (default) Embedding =========================================================')
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
	data_names = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']
	for data_name in data_names:
		encoder_name = get_encoder_name('AE', data_name, mark='NoEarlyStop')
		config = AEConfig(); config.ae_acts[-1] = None
		# config.ae_drop_out_rate = 0.0
		config.early_stop_patience = None
		evaluator.run_all_embedding(
			data_name, encoder_name, 'AE', embed_repeat=embed_repeat, data_type=DATA_MAT, config=config)
		evaluator.eval_embedding(data_name, encoder_name, embed_repeat=embed_repeat)


	# monitor memory:  mprof run --python -C python dim_reduce_clt_eval.py && mprof plot -o mem.png -t "Memory Use"
	print('PCA/MDS Timing =========================================================')
	embed_repeat = 3
	embed_methods = ['PCA']
	reduced_dim = 256
	data_names = get_all_sample_data_names('1M_neurons', [1000, 5000, 10000, 50000, 100000, 300000])
	for embed_method, data_name in itertools.product(embed_methods, data_names):
		encoder_name = get_encoder_name(embed_method, data_name, reduced_dim, mark='Default-Mem')
		evaluator.run_all_embedding(
			data_name, encoder_name, embed_method, embed_repeat=embed_repeat, data_type=DATA_MAT, dim=reduced_dim)
		evaluator.combine_time_result(encoder_name, embed_repeat)


	print('AIDE Timing =========================================================')
	embed_repeat = 3
	data_names = get_all_sample_data_names('1M_neurons', [1000, 5000, 10000, 50000, 100000, 300000, 500000, 1000000]) + ['1M_neurons']
	data_types = [DATA_MAT, DATA_MAT, DATA_MAT, DATA_TFRECORD, DATA_TFRECORD, DATA_TFRECORD, DATA_TFRECORD,DATA_TFRECORD]
	for data_name, data_type in zip(data_names, data_types):
		encoder_name = get_encoder_name('AIDE', data_name, mark='Default')
		config = AIDEConfig()
		evaluator.run_all_embedding(
			data_name, encoder_name, 'AIDE', embed_repeat=embed_repeat, data_type=data_type, config=config, save_embedding=False)
		evaluator.combine_time_result(encoder_name, embed_repeat)


	print('K - ARI/NMI =========================================================')
	data_names = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']
	default_clt_config_dict = {data_name: {} for data_name in data_names}
	embed_configs = [
		('PCA', '', RPKMeans, PCA_RP_KMEANS_CONFIG_DICT, 'rp-kmeans (best; 10)'),
		('PCA', '', RPKMeans, default_clt_config_dict, 'rp-kmeans (default; 10)'),
		('PCA', '', KMeans, default_clt_config_dict, 'kmeans (k-means++; 10)'),
		('PCA', '', KMeans, {dn:{'init':'random'} for dn in data_names}, 'kmeans (random; 10)'),
		('AIDE', 'Best', RPKMeans, AIDE_RP_KMEANS_CONFIG_DICT, 'rp-kmeans (best; 10)'),
		('AIDE', 'Best', RPKMeans, default_clt_config_dict, 'rp-kmeans (default; 10)'),
		('AIDE', 'Best', KMeans, default_clt_config_dict, 'kmeans (k-means++; 10)'),
		('AIDE', 'Best', KMeans, {dn:{'init':'random'} for dn in data_names}, 'kmeans (random; 10)'),
	]
	for embed_method, mark, clt_initializer, dn_to_clt_kwargs, clt_name in embed_configs:
		for data_name in data_names:
			encoder_name = get_encoder_name(embed_method, data_name, mark=mark)
			clt_kwargs = dn_to_clt_kwargs[data_name]; clt_kwargs['n_init'] = 10; clt_kwargs['verbose'] = 0
			if embed_method == 'AIDE' and data_name == 'mouse_bladder_cell' and clt_name.find('default') != -1:
				clt_kwargs['w'] = 0.5
			evaluator.run_k_metric(data_name, encoder_name,
				embed_repeat_id=0, clt_name=clt_name, clt_initializer=clt_initializer, clt_kwargs=clt_kwargs)

	data_names = ['deng', 'llorens']
	default_clt_config_dict = {data_name: {} for data_name in data_names}
	embed_configs = [
		('PCA', '', RPKMeans, default_clt_config_dict, 'rp-kmeans (default; 10)'),
		('PCA', '', KMeans, default_clt_config_dict, 'kmeans (k-means++; 10)'),
		('PCA', '', KMeans, {dn:{'init':'random'} for dn in data_names}, 'kmeans (random; 10)'),
		('AIDE', 'Default', RPKMeans, default_clt_config_dict, 'rp-kmeans (default; 10)'),
		('AIDE', 'Default', KMeans, default_clt_config_dict, 'kmeans (k-means++; 10)'),
		('AIDE', 'Default', KMeans, {dn:{'init':'random'} for dn in data_names}, 'kmeans (random; 10)'),
	]
	for embed_method, mark, clt_initializer, dn_to_clt_kwargs, clt_name in embed_configs:
		for data_name in data_names:
			dim = 128 if embed_method == 'PCA' and data_name == 'llorens' else 256
			encoder_name = get_encoder_name(embed_method, data_name, mark=mark, dim=dim)
			clt_kwargs = dn_to_clt_kwargs[data_name]; clt_kwargs['n_init'] = 10; clt_kwargs['verbose'] = 0
			evaluator.run_k_metric(data_name, encoder_name,
				embed_repeat_id=0, clt_name=clt_name, clt_initializer=clt_initializer, clt_kwargs=clt_kwargs)

	data_names = get_all_sim_dropout_data_names([60, 70, 75, 80])
	default_clt_config_dict = {data_name:{} for data_name in data_names}
	embed_configs = [
		('PCA', '', RPKMeans, default_clt_config_dict, 'rp-kmeans (default; 10)'),
		('PCA', '', KMeans, default_clt_config_dict, 'kmeans (k-means++; 10)'),
		('PCA', '', KMeans, {dn:{'init':'random'} for dn in data_names}, 'kmeans (random; 10)'),
		('AIDE', 'Default-40000', RPKMeans, default_clt_config_dict, 'rp-kmeans (default; 10)'),
		('AIDE', 'Default-40000', KMeans, default_clt_config_dict, 'kmeans (k-means++; 10)'),
		('AIDE', 'Default-40000', KMeans, {dn:{'init':'random'} for dn in data_names}, 'kmeans (random; 10)'),
	]
	for embed_method, mark, clt_initializer, dn_to_clt_kwargs, clt_name in embed_configs:
		for data_name in data_names:
			encoder_name = get_encoder_name(embed_method, data_name, mark=mark)
			clt_kwargs = dn_to_clt_kwargs[data_name]; clt_kwargs['n_init'] = 10; clt_kwargs['verbose'] = 0
			evaluator.run_k_metric(data_name, encoder_name,
				embed_repeat_id=0, clt_name=clt_name, clt_initializer=clt_initializer, clt_kwargs=clt_kwargs)


	print('K Selection =========================================================')
	data_names = ['10X_PBMC', 'mouse_bladder_cell', 'mouse_ES_cell', 'worm_neuron_cell', 'Shekhar_mouse_retina', 'PBMC_68k', 'sc_brain']
	embed_configs = [('PCA', ''), ('AIDE', 'Best')]
	for embed_method, mark in embed_configs:
		for data_name in data_names:
			encoder_name = get_encoder_name(embed_method, data_name, mark=mark)
			point_reducer_kwargs = {'verbose': 0}
			if embed_method == 'AIDE' and data_name == 'mouse_bladder_cell': point_reducer_kwargs['w'] = 0.5
			evaluator.run_k_selection(data_name, encoder_name, embed_repeat_id=0, point_reducer_kwargs=point_reducer_kwargs)

	data_names = ['deng', 'llorens']
	embed_configs = [('PCA', ''), ('AIDE', 'Default')]
	for embed_method, mark in embed_configs:
		for data_name in data_names:
			dim = 128 if embed_method == 'PCA' and data_name == 'llorens' else 256
			encoder_name = get_encoder_name(embed_method, data_name, mark=mark, dim=dim)
			evaluator.run_k_selection(data_name, encoder_name, embed_repeat_id=0, point_reducer_kwargs={'verbose': 0})

	data_names = get_all_sim_dropout_data_names([60, 70, 75, 80])
	embed_configs = [('PCA', ''), ('AIDE', 'Default-40000')]
	for embed_method, mark in embed_configs:
		for data_name in data_names:
			encoder_name = get_encoder_name(embed_method, data_name, mark=mark)
			evaluator.run_k_selection(data_name, encoder_name, embed_repeat_id=0, point_reducer_kwargs={'verbose': 0})


