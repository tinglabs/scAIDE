"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import matplotlib.pyplot as plt
plt.switch_backend('agg')
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import pandas as pd


def simple_dist_plot(figpath, x, bins, x_label=None, title=None, figsize=None, x_lim=(None, None)):
	x_label = 'x' if x_label is None else x_label
	title = 'Simple Dist Plot' if title is None else title
	plt.figure(figsize=figsize)
	ax = plt.axes()
	sns.distplot(x, bins=bins, kde=False, rug=False, axlabel=x_label, ax=ax)
	ax.set_title(title)
	ax.set_xlim(left=x_lim[0], right=x_lim[1])
	plt.savefig(figpath)
	plt.close()


def simple_dot_plot(figpath, x, y, p_types=None, p_type_order=None, sizes=None, markers=None, p_id_to_text=None,
		x_label=None, y_label=None, title=None, figsize=None, p_type_label=None, p_size_label=None, p_style_label=None,
		palette=None):
	"""
	Args:
		x (array-like)
		y (array-like)
		p_types (array-like): different point color for different pType
		p_type_order (larray-like):
		sizes (dict or list or tuple or None): {label: size}
		markers (dict or list or None): {label: marker}
		p_id_to_text (dict): {pId: str}
	"""
	x_label = 'x' if x_label is None else x_label
	y_label = 'y' if y_label is None else y_label
	title = 'Simple Dot Plot' if title is None else title
	p_id_to_text = {} if p_id_to_text is None else p_id_to_text
	if palette is None and p_types is not None:
		palette = sns.color_palette("hls", len(set(p_types)))

	df_dict = {x_label: x, y_label: y}
	if p_types is not None:
		df_dict[p_type_label] = p_types
	df = pd.DataFrame(df_dict)

	plt.figure(figsize=figsize)
	ax = plt.axes()
	fig = sns.scatterplot(x=x_label, y=y_label, hue=p_type_label, size=p_size_label, style=p_style_label,
		hue_order=p_type_order, sizes=sizes, markers=markers, data=df, ax=ax, palette=palette)
	for pId, pText in p_id_to_text.items():
		fig.text(x[pId]+0.02, y[pId], pText, horizontalalignment='left', size='medium', color='black', weight='semibold')
	ax.set_title(title)

	plt.savefig(figpath)
	plt.close()





