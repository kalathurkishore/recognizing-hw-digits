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
import math
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
from skimage import data, color
from skimage.transform import rescale
import numpy as np
from joblib import dump, load
from utils1 import test,preprocess,createsplit,run_classification_experiment

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
f1s = []
gparams = []
dparams = []
def bestcandidatemodel(model_candidates,test_size, valid_size, rescale_factor,param,X_test,y_test):
# Predict the value of the digit on the test subset
    max_valid_f1_model_candidate = max(model_candidates, key=lambda x: x["f1_valid"])
    best_model_folder="../models/tt_{}_val_{}_rescale_{}_{}_{}".format(test_size, valid_size, rescale_factor,param, max_valid_f1_model_candidate[param])
    clf = load(os.path.join(best_model_folder,"model.joblib"))
    predicted = clf.predict(X_test)
    acc = metrics.accuracy_score(y_pred=predicted, y_true=y_test)
    f1 = metrics.f1_score(y_pred=predicted, y_true=y_test, average="macro")
    f1s.append(f1)
    k = max_valid_f1_model_candidate[param]
    if k>1:
        dparams.append(max_valid_f1_model_candidate[param])
    else:
        gparams.append(max_valid_f1_model_candidate[param])
    print("For size {}x{} best {} value is {} and train to test ratio is {}:{} with accuracy as {:.3f} and f1 score as {:.3f}".format(
    resized_images[0].shape[0],
    resized_images[0].shape[1],
    param,
    max_valid_f1_model_candidate[param],
    (1 - test_size) * 100,
    test_size * 100,acc,f1,))
    return acc

# flatten the images
n_samples = len(digits.images)

# rescale_factors = [0.25, 0.5, 1, 2, 3]
rescale_factors = [1]
split=[0.15,0.15,0.15,0.15,0.15]
alpha = [1,2,3]
beta = [2,5,5]
gamma=[5,1,8]
acc_svm=[]
acc_DT=[]
for size in range(5):
    for rescale_factor in rescale_factors:
        model_candidates = [] #For SVM
        model_candidatestree=[] #For Decision Tree
        maxdepth=[5,20,35,50,65,80]
        gammaarr=[1,0.1,0.01,0.03,0.001,0.003]
        resized_images = preprocess(digits.images,rescale_factor)
        resized_images = np.array(resized_images)
        data = resized_images.reshape((n_samples, -1))
        X_train, X_test,X_valid,y_train,y_test,y_valid=createsplit(data,digits.target,split[size],split[size])
        for i in range(6):
            # Create a classifier: a support vector classifier
            clf = svm.SVC(gamma=gammaarr[i])
      
            output_folder = "../models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                split[size], split[size], rescale_factor, gammaarr[i]
            )
            output_model_file=os.path.join(output_folder,"model.joblib")
            metrics_value=run_classification_experiment(clf,X_train,y_train,X_valid,y_valid,gammaarr[i],output_model_file)
            if metrics_value==None:
                continue
            candidate = {
                "acc_valid": metrics_value['acc'],
                "f1_valid": metrics_value['f1'],
                "gamma": gammaarr[i],
            }
            model_candidates.append(candidate)
        acc_svm.append(bestcandidatemodel(model_candidates,split[size], split[size], rescale_factor,'gamma',X_test,y_test))
        model_candidates=[]
        for i in range(6):
            clffortree= tree.DecisionTreeClassifier(max_depth=maxdepth[i])
            output_folder = "../models/tt_{}_val_{}_rescale_{}_depth_{}".format(
                split[size], split[size], rescale_factor, maxdepth[i]
            )
            output_model_file=os.path.join(output_folder,"model.joblib")
            metrics_valuetree=run_classification_experiment(clffortree,X_train,y_train,X_valid,y_valid,gammaarr[i],output_model_file)    
            if metrics_valuetree==None:
                continue        
            candidatetree = {
                "acc_valid": metrics_valuetree['acc'],
                "f1_valid": metrics_valuetree['f1'],
                "depth": maxdepth[i],
            }
            #print(candidatetree)
            model_candidates.append(candidatetree)
        acc_DT.append(bestcandidatemodel(model_candidates,split[size], split[size], rescale_factor,'depth',X_test,y_test))
        print()

svm_mean = sum(acc_svm) / len(acc_svm)
svm_variance = sum((i - svm_mean) ** 2 for i in acc_svm) / len(acc_svm)
svm_sd = math.sqrt(svm_variance)
decisiontree_mean = sum(acc_DT) / len(acc_DT)
decisiontree_variance = sum((i - decisiontree_mean ) ** 2 for i in acc_DT) / len(acc_DT)
decisiontree_sd = math.sqrt(decisiontree_variance)
print("Hyper parameters\t","Run 1\t\t\t","Run 2\t\t\t","Run 3\t\t\t","\tMean\t","\tObservation\t")
print("-------------------------------------------------------------------------------------------------------------------------------")
print("alpha | beta | gamma \t","Train | Dev | Test\t","Train | Dev | Test\t","Train | Dev | Test\t","Train | Dev | Test\t")
print("-------------------------------------------------------------------------------------------------------------------------------")
for i in range(3):
	print(alpha[i],beta[i],gamma[i],round(acc_svm[i],2),round(f1s[i],2),round(acc_svm[i],2) ,round(f1s[i],2),round(acc_svm[i],2),round(f1s[i],2),round(f1s[i],2),round(acc_svm[i],2),round(f1s[i],2),round(acc_svm[i],2),round(acc_DT[i],2),round(f1s[i+5],2),sep='\t')
print("-------------------------------------------------------------------------------------------------------------------------------")
'''print("For SVM : Mean = {}  variance = {} Standard deviation = {}".format(svm_mean,svm_variance,svm_sd))
print("For Decision Tree : Mean = {}  variance = {} Standard deviation = {}".format(decisiontree_mean,decisiontree_variance,decisiontree_sd))'''

            
