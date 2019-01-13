from sklearn.cluster import MiniBatchKMeans

import skimage.io as io
import numpy as np
import os, sys
import json
import subprocess as sp

from skimage.morphology import disk
from skimage.morphology import dilation

def cloudmasking(self,):

    band = os.path.join(self.workdir, band + ".tif")

    img_ds = io.imread(band)
    img = np.array(img_ds, dtype='uint16')

    row = img.shape[0]
    col = img.shape[1]

    ras_shape = (row * col, 1)
    img_as_array = img[:, :].reshape(ras_shape)

    cluster = MiniBatchKMeans(n_clusters=2,
                              tol=0.00001,
                              n_init=10,
                              init='random',
                              max_no_improvement=100,
                              reassignment_ratio=0.05)

    cluster_labels = cluster.fit_predict(img_as_array)
    cluster_center = cluster.cluster_centers_

    new_shape = (row, col)
    img_clusters = cluster_labels.reshape(new_shape)

    centroids_num = len(cluster_center)

    liste = []
    for i in range(0, centroids_num):

        means = np.mean(cluster_center[i])
        liste.append(means)

    max = np.max(liste)
    val = liste.index(max)

    if max > 1800:

        mask = np.where(img_clusters == val, 1, np.nan)
        img_mask = np.array(mask, dtype='uint8').reshape((row, col))

        kernel = disk(5)
        img_buffer = dilation(img_mask, kernel)

        cloudmask = os.path.join(self.workdir, "cloudmask.tif").format(sys.platform, os.sep)
        io.imsave(cloudmask, img_buffer)    