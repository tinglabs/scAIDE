"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import itertools
import time
import os

from aide.utils_ import get_load_func, get_save_func


def load_save(attrPath, file_format):
	def outer_wrapper(func):
		def wrapper(cls, *args, **kwargs):
			path = getattr(cls, attrPath)
			if os.path.exists(path):
				load_func = get_load_func(file_format)
				coll = load_func(path)
				return coll
			coll = func(cls, *args, **kwargs)
			saveFunc = get_save_func(file_format)
			saveFunc(coll, path)
			return coll
		return wrapper
	return outer_wrapper


def dict_value_to_list(d):
	ret = {}
	for k, v in d.items():
		if isinstance(v, list) or isinstance(v, tuple):
			ret[k] = v
		else:
			ret[k] = [v]
	return ret


def unzip_dict(d):
	"""
	Args:
		d (dict): e.g. {
			k1: [v1, v2],
			k2: v3,
			k3: [v4, v5]
		}
	Returns:
		list: e.g. [
			{k1: v1, k2: v3, k3: v4},
			{k1: v1, k2: v3, k3: v5},
			{k1: v2, k2: v3, k3: v4},
			{k1: v2, k2: v3, k3: v5}
		]
	"""
	if len(d) == 0:
		return []
	d = dict_value_to_list(d)
	ret_list = []
	k_list, v_lists = zip(*d.items())
	for v_list in itertools.product(*v_lists):
		ret_list.append({k: v for k, v in zip(k_list, v_list)})
	return ret_list


if __name__ == '__main__':
	print(unzip_dict({'k1': ['v1', 'v2'], 'k2': 'v3', 'k3': ['v4', 'v5']}))

