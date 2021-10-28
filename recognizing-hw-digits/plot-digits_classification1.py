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
def bestcandidatemodel(model_candidates,test_size, valid_size, rescale_factor,param,X_test,y_test):
# Predict the value of the digit on the test subset
    max_valid_f1_model_candidate = max(model_candidates, key=lambda x: x["f1_valid"])
    best_model_folder="../models/tt_{}_val_{}_rescale_{}_{}_{}".format(test_size, valid_size, rescale_factor,param, max_valid_f1_model_candidate[param])
    clf = load(os.path.join(best_model_folder,"model.joblib"))
    predicted = clf.predict(X_test)
    acc = metrics.accuracy_score(y_pred=predicted, y_true=y_test)
    f1 = metrics.f1_score(y_pred=predicted, y_true=y_test, average="macro")
    print("For size {}x{} best {} value is {} train to test ratio is {}:{} with accuracy as {:.3f} and f1 score as {:.3f}".format(
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
split=[0.05,0.1,0.15,0.2,0.25]
acc_svm=[]
acc_DT=[]
for size in range(5):
    for rescale_factor in rescale_factors:
        model_candidates = []
        model_candidatestree=[]
        maxdepth=[5,20,35,50,65,80]
        gammaarr=[1,0.5,0.01,0.001,0.0001,0.000005]
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
decisiontree_mean = sum(acc_DT) / len(acc_DT)
decisiontree_variance = sum((i - decisiontree_mean ) ** 2 for i in acc_DT) / len(acc_DT)
print("For SVM : Mean = {}  variance = {} ".format(svm_mean,svm_variance))
print("For Decision Tree : Mean = {}  variance = {} ".format(decisiontree_mean,decisiontree_variance))

            
'''
# flatten the images
n_samples = len(digits.images)

# rescale_factors = [0.25, 0.5, 1, 2, 3]
rescale_factors = [1]
for test_size, valid_size in [(0.15, 0.15)]:
    for rescale_factor in rescale_factors:
        model_candidates = []
        for gamma in [1,0.5,0.01,0.001,0.0001,0.000005]:
            resized_images = preprocess(digits.images,rescale_factor)

            resized_images = np.array(resized_images)
            data = resized_images.reshape((n_samples, -1))

            # Create a classifier: a support vector classifier
            clf = svm.SVC(gamma=gamma)

            clf1 = tree.DecisionTreeClassifier()
            output_folder = "../models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, valid_size, rescale_factor, gamma
            )
            output_model_file=os.path.join(output_folder,"model.joblib")

            X_train, X_test,X_valid,y_train,y_test,y_valid=createsplit(data,digits.target,test_size,valid_size)

            # print("Number of samples: Train:Valid:Test = {}:{}:{}".format(len(y_train),len(y_valid),len(y_test)))
            metrics_value=run_classification_experiment(clf,X_train,y_train,X_valid,y_valid,gamma,output_model_file)
            metrics_value=run_classification_experiment1(clf1,X_train,y_train,X_valid,y_valid,gamma,output_model_file)

            # Learn the digits on the train subset
            #print(metrics_value)
            
            # we will ensure to throw away some of the models that yield random-like performance.
            if metrics_value==None:
                continue

            candidate = {
                "acc_valid": metrics_value['acc'],
                "f1_valid": metrics_value['f1'],
                "gamma": gamma,
            }
            model_candidates.append(candidate)
            
            #os.mkdir(output_folder)
            #dump(clf, os.path.join(output_folder,"model.joblib"))

            
        # Predict the value of the digit on the test subset

        max_valid_f1_model_candidate = max(
            model_candidates, key=lambda x: x["f1_valid"]
        )
        best_model_folder="../models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, valid_size, rescale_factor, max_valid_f1_model_candidate['gamma']
            )
        clf = load(os.path.join(best_model_folder,"model.joblib"))
        predicted = clf.predict(X_test)
        predicted1 = clf1.predict(X_test)
        acc = metrics.accuracy_score(y_pred=predicted, y_true=y_test)
        f1 = metrics.f1_score(y_pred=predicted, y_true=y_test, average="macro")
        acc1 = metrics.accuracy_score(y_pred=predicted1, y_true=y_test)
        f11 = metrics.f1_score(y_pred=predicted1, y_true=y_test, average="macro")
        print(
            "For SVM size {}x{} the best gamma value is {} train to test ratio is {}:{} with accuracy as {:.3f} and f1 score as {:.3f}".format(
                resized_images[0].shape[0],
                resized_images[0].shape[1],
                max_valid_f1_model_candidate["gamma"],
                (1 - test_size) * 100,
                test_size * 100,
                acc,
                f1,
            )
        )
        print(
            "For Decision Tree size {}x{} the best gamma value is {} train to test ratio is {}:{} with accuracy as {:.3f} and f1 score as {:.3f}".format(
                resized_images[0].shape[0],
                resized_images[0].shape[1],
                max_valid_f1_model_candidate["gamma"],
                (1 - test_size) * 100,
                test_size * 100,
                acc1,
                f11,
            )
        )'''