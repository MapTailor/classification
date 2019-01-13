import skimage.io as io
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedShuffleSplit
import numpy.ma as ma
import os, shutil

def Classification():

	raster = path_to_tiff + ".tif"
	samples = path_to_tiff + "_samples.tif"
	tfw_old = path_to_tiff + ".tfw"

	classification = rootdir + "Classification\\" + pathrow + "_" + year + ".tif"
					
	# read Landsat data as TIF
	img_ds = io.imread(raster)
	img = np.array(img_ds, dtype='uint16')

	# read training samples as TIF with same dimensions as the Landsat image
	roi_ds = io.imread(samples)
	roi = np.array(roi_ds, dtype='uint8')

	labels = np.unique(roi[roi > 0])
	print('The training data include {n} classes: {classes}'.format(n=labels.size, classes=labels))

	X = img[roi > 0, :]
	Y = roi[roi > 0]

	# splitting of training & test data in 80% - 20% for outlier analysis
	sss = StratifiedShuffleSplit(train_size=0.8, n_splits=2, test_size=0.2, random_state=None)
	sss.get_n_splits(X, Y)

	for train_index, test_index in sss.split(X, Y):
	    X_train, X_test = X[train_index], X[test_index]
	    Y_train, Y_test = Y[train_index], Y[test_index]

	gbt = GradientBoostingClassifier(n_estimators = 100,
					min_samples_leaf = 4,
					min_samples_split = 10,
					max_depth = 10,
					max_features = 'sqrt',
					learning_rate = 0.01,
					subsample = 0.8,
					random_state = None,
					warm_start = False).fit(X_train, Y_train)

	svm = SVC(C = 1.0,
		kernel = 'rbf',
		gamma = 1.0,
		tol = 0.005,
		probability = True,
		class_weight = 'balanced',
		cache_size = 8000,
		decision_function_shape = 'ovr',
		max_iter = 1000).fit(X_train, Y_train)

	# Voting classifier for Gradient Boosting and SVM
	clf = VotingClassifier(estimators=[('gbt', gbt), ('svm', svm)], voting = 'soft', weights = [1,1]).fit(X_train, Y_train)

	# Feature Importances of the Gradient Boosting classifier
	print gbt.feature_importances_

	# save the classification model
	model = rootdir + "MODEL\\" + pathrow + "_" + year + ".pkl"
	joblib.dump(clf, model)

	# call the classification model
	clf = joblib.load(model)

	row = img.shape[0]
	col = img.shape[1]
	dim = img.shape[2]

	# reshaping of the array with 10 features/bands
	new_arr = (row * col, dim)
	new_img = img[:, :, :dim].reshape(new_arr)

	# calculating classification probability, e.g. in this case with 7 classes
	class_estimation = clf.predict_proba(new_img)

	for n in range(0,7):
		idx = str(n + 1)
		class_prob = prob[:, n]
		class_prob = class_prob.reshape((row, col))

		probability = rootdir + "Probability\\" + pathrow + "_" + year + "_" + idx + ".tif"
		io.imsave(probability, class_prob)

		tfw_new = probability.split(".tif")[0] + ".tfw"
		shutil.copy(tfw_old, tfw_new)

	# saving the classification results
	class_prediction = clf.predict(new_img)
	class_prediction = class_prediction.reshape(img[:, :, 0].shape)

	io.imsave(classification, class_prediction)

	tfw_new = classification.split(".tif")[0] + ".tfw"
	shutil.copy(tfw_old, tfw_new)

