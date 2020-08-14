"""
@Author: Yu Huang
@Email: yuhuang-cst@foxmail.com
"""

import numpy as np

from preprocess import BasicPreprocessor
from dist_ae import DistAutoEncoder, DistAutoEncoderConfig
from rph_kmeans import RPHKMeans

print('get data and preprocess')
X = np.random.rand(1000, 2000)  # (n_samples, n_features)
X, _ = BasicPreprocessor('test').process_(X, labels=None)

print('dimension reduction')
config = DistAutoEncoderConfig()
config.max_epoch_num = 300
encoder = DistAutoEncoder(name='TestEncoder', save_folder='TestFolder')
embedding = encoder.fit_transform(X, config=config)	# (n_samples, 256)

print('clustering')
clt = RPHKMeans()
y_pred = clt.fit_predict(embedding) # (n_samples,)



