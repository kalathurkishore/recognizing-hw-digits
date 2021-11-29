"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import test, preprocess,createsplitwithsuffle,run_classification_experiment,train_val_splits

from sklearn.metrics import plot_confusion_matrix
from sklearn import datasets, svm, metrics
from sklearn.tree import DecisionTreeClassifier
from joblib import dump, load



plt.rcParams.update({'figure.max_open_warning': 0})

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
train_split = [10,20,30,40,50,60,70,80,90,100]

svm_columns = ['Training Data', 'Gamma', 'SVM_Test_Accuracy', 'SVM_Validation_Accuracy', 'SVM_F1_Score']
svm_output = pd.DataFrame(data = [], columns=svm_columns)
gamma = [1,0.1,0.01,0.03,0.001,0.003]

dt_columns = ['Trainining Data', 'MaxDepth', 'DT_Test_Accuracy',  'DT_Validation_Accuracy', 'DT_F1_Score']
dt_output = pd.DataFrame(data = [], columns=dt_columns)
depths = [5,12,20,35,50,65]


def dt_train(x_train, y_train, x_val, y_val, x_test, y_test, depth, cmd=False, td=None):
    dt = DecisionTreeClassifier(max_depth=depth)
    t_ac,val_ac,predicted,f1=train_val_splits(dt,x_train,y_train,x_test, y_test,x_val, y_val)
    if cmd:
        cm = metrics.confusion_matrix(predicted, y_test, labels  = [0,1,2,3,4,5,6,7,8,9])
        disp = metrics.ConfusionMatrixDisplay(cm)
        ttl = 'DT Confusion Matrix for ' + str(td) + '% training data'
        disp.plot()
        plt.title(ttl)
        file_name = "%s.png" % ttl
        plt.savefig(file_name)
        plt.show()
    return t_ac, val_ac, f1

def svm_train(x_train, y_train, x_val, y_val, x_test, y_test, gamma, cmd=False, td = None):
    clf = svm.SVC(gamma=gamma)
    t_ac,val_ac,predicted,f1=train_val_splits(clf,x_train,y_train,x_test, y_test,x_val, y_val)
    plot_confusion_matrix(clf, x_train, y_train)  
    plt.show()
    if cmd:
        cm = metrics.confusion_matrix(predicted, y_test, labels  = [0,1,2,3,4,5,6,7,8,9])
        disp = metrics.ConfusionMatrixDisplay(cm)
        ttl = 'SVM Confusion Matrix for ' + str(td) + '% training data'
        disp.plot()
        plt.title(ttl)
        file_name = "%s.png" % ttl
        plt.savefig(file_name)
    return t_ac, val_ac, f1
test_size=0.1
valid_size=0.1
resized_images = preprocess(digits.images,1)
resized_images = np.array(resized_images)
data = resized_images.reshape((n_samples, -1))
x_train, x_test,x_val,y_train,y_test,y_val = createsplitwithsuffle(data, digits.target, test_size, valid_size)
for gamma in gamma:
  for tr in train_split:
    sp = int(tr/100 * len(x_train))
    n_train = x_train[:sp]
    n_ytrain = y_train[:sp]
    if gamma == 0.001:
      st_ac, sval_ac, sf1 = svm_train(n_train, n_ytrain, x_val, y_val, x_test, y_test, gamma, True, tr)
    else:
      st_ac, sval_ac, sf1 = svm_train(n_train, n_ytrain, x_val, y_val, x_test, y_test, gamma)
    out = pd.DataFrame(data = [[tr, gamma, st_ac, sval_ac, sf1]],columns = svm_columns)
    svm_output = svm_output.append(out, ignore_index=True)
    
for depth in depths:
  for tr in train_split:
    sp = int(tr/100 * len(x_train))
    n_train = x_train[:sp]
    n_ytrain = y_train[:sp]
    if depth == 12:
      t_ac, val_ac, f1 = dt_train(n_train, n_ytrain, x_val, y_val, x_test, y_test, depth, True, tr)
    else:
      t_ac, val_ac, f1 = dt_train(n_train, n_ytrain, x_val, y_val, x_test, y_test, depth)
    out = pd.DataFrame(data = [[tr, depth, t_ac, val_ac, f1]],
    columns = dt_columns)
    dt_output = dt_output.append(out, ignore_index=True)

print("SVM Training Output for splits ")
print(svm_output)
svm_output.to_csv("SVM_output.csv")
print("Decision Tree Training Output for splits ")
print(dt_output)
dt_output.to_csv("DT_output.csv")