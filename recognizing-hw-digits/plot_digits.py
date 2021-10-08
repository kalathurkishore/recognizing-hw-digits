"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9. 
"""

#print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import os
import matplotlib.pyplot as plt
import glob
import sys 

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm
from skimage import data, color
import numpy as np

from utils1 import preprocess, create_splits, test_
from joblib import dump, load


base_path = '/home/kishore/hwdigits/recognizing-hw-digits/recognizing-hw-digits'        
sys.path.append(base_path)

digits = datasets.load_digits()
n_samples = len(digits.images)

def classification_of_data(digits, isTrain = False):

    rescale_factors = [1]
    for test_size, valid_size in [(0.15, 0.15)]:
        for rescale_factor in rescale_factors:
            model_candidates = []
            for gamma in [10 ** exponent for exponent in range(-7, 0)]:
                resized_images = preprocess(
                    images=digits.images, rescale_factor=rescale_factor
                )
                resized_images = np.array(resized_images)
                data = resized_images.reshape((n_samples, -1))

                # Create a classifier: a support vector classifier
                clf = svm.SVC(gamma=gamma)
                
                X_train, X_test, X_valid, y_train, y_test, y_valid = create_splits(
                    data, digits.target, test_size, valid_size
                )

                if isTrain:
                    X_test = X_train = X_valid = data
                    y_train = y_valid = y_test = digits.target
                    

                clf.fit(X_train, y_train)
                metrics_valid = test_(clf, X_valid, y_valid)
                
                # we will ensure to throw away some of the models that yield random-like performance.
                if metrics_valid['acc'] < 0.11:
                    print("Skipping for {}".format(gamma))
                    continue

                candidate = {
                    "acc_valid": metrics_valid['acc'],
                    "f1_valid": metrics_valid['f1'],
                    "gamma": gamma,
                }
                model_candidates.append(candidate)
                if isTrain:
                    output_folder = base_path+"/models/tt_{}_val_{}_rescale_{}_gamma_{}_train".format(
                    test_size, valid_size, rescale_factor, gamma)
                else :
                    output_folder = base_path+"/models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                    test_size, valid_size, rescale_factor, gamma)
            
                os.mkdir(output_folder)
                dump(clf, os.path.join(output_folder, "model.joblib"))

            # Predict the value of the digit on the test subset

            max_valid_f1_model_candidate = max(
                model_candidates, key=lambda x: x["f1_valid"]
            )
            best_model_folder = base_path+ "/models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, valid_size, rescale_factor, max_valid_f1_model_candidate["gamma"]
            )

            best_model_file = "/models/tt_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, valid_size, rescale_factor, max_valid_f1_model_candidate["gamma"]
            )
            clf = load(os.path.join(best_model_folder, "model.joblib"))

            metrics = test_(clf, X_test, y_test)
            print(
                "{}x{}\t{}\t{}:{}\t{:.3f}\t{:.3f}".format(
                    resized_images[0].shape[0],
                    resized_images[0].shape[1],
                    max_valid_f1_model_candidate["gamma"],
                    (1 - test_size) * 100,
                    test_size * 100,
                    metrics['acc'],
                    metrics['f1'],
                )
            )
            return metrics,best_model_file
