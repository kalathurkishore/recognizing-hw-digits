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
import matplotlib.pyplot as plt

import os

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

from skimage import data, color 
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np

from joblib import dump,load

from utils import preprocess, test , create_splits
'''
def preprocess(images,rescale_factor):
   resized_images = []
   for d in digits.images:
      resized_images.append(rescale(d,rescale_factor,anti_aliasing=False))
   return resized_images

def create_splits(data, targets, test_size,validation_size):
   X_train, X_test_valid, y_train, y_test_valid = train_test_split(data, targets, test_size=test_size + validation_size, shuffle=False)
   X_test, X_validate, y_test, y_validate = train_test_split(X_test_valid, y_test_valid, test_size=validation_size/(test_size+validation_size), shuffle=False)
   return X_train,X_test,X_validate,y_train,y_test,y_validate

def test(clf,X,y):
   predicted = clf.predict(X)
   accuracy = metrics.accuracy_score(y_pred = predicted,y_true = y)
   f1 = metrics.f1_score(y_pred = predicted,y_true = y,average = "macro")

   return {'acc':acc,'f1':f1}
'''

digits = datasets.load_digits()

# flatten the images
n_samples = len(digits.images)
rescale_factors = [1]
for test_size,validation_size in [(0.15,0.15)]:
   for rescale_factor in rescale_factors:
      model_candidates = []
      for gamma in [10 ** exponent for exponent in range(-7,0)]:
         resized_images = preprocess(digits.images,rescale_factor=rescale_factor)
         '''for d in digits.images:
             resized_images.append(rescale(d,rescale_factor,anti_aliasing=False))'''
         resized_images = np.array(resized_images)
         data = resized_images.reshape((n_samples, -1))

         clf = svm.SVC(gamma=gamma)

         X_train, X_test, X_validate, y_train, y_test, y_validate = create_splits(data, digits.target, test_size,validation_size)
         '''X_train, X_test_valid, y_train, y_test_valid = train_test_split(
         data, digits.target, test_size=test_size + validation_size, shuffle=False)
         X_test, X_validate, y_test, y_validate = train_test_split(
         X_test_valid, y_test_valid, test_size=validation_size/(test_size+validation_size), shuffle=False)'''
         clf.fit(X_train, y_train)
         metrics_validate = test(clf,X_validate,y_validate)
         '''clf.fit(X_train, y_train)
         predicted_validate_set = clf.predict(X_validate)
         accuracy_valid = metrics.accuracy_score(y_pred = predicted_validate_set,y_true = y_validate)
         f1_valid = metrics.f1_score(y_pred = predicted_validate_set,y_true = y_validate,average = "macro")'''
         if metrics_validate['acc'] < 0.11:
            print("skipping for {}",format(gamma))
            continue
         candidate = {
            "accuracy_valid": metrics_validate['acc'],
            "f1_valid": metrics_validate['f1'],
            "gamma": gamma,
         }
         model_candidates.append(candidate)
         output_folder = "../models/tt_{}_val_{}_rescale_{}_gamma_{}.joblib".format(test_size, validation_size, rescale_factor, gamma)
         os.mkdir(output_folder)
         dump(clf,os.path.join(output_folder,"model.joblib"))

      max_valid_f1_model_candidate = max(model_candidates,key=lambda x: x["f1_valid"])
      best_model_folder = "../models/tt_{}_val_{}_rescale_{}_gamma_{}.joblib".format(test_size, validation_size, rescale_factor, max_valid_f1_model_candidate['gamma'])
      clf = load(os.path.join(best_model_folder,"model.joblib"))
      predicted = clf.predict(X_test)
      accuracy = metrics.accuracy_score(y_pred = predicted,y_true = y_test)
      f1 = metrics.f1_score(y_pred = predicted,y_true = y_test,average = "macro")
     #metrics = test(clf,X_test,y_test)
      print("{}*{}\t{}\t{}:{}\t{:.5f}\t{:.5f}".format(resized_images[0].shape[0],resized_images[0].shape[1],max_valid_f1_model_candidate['gamma'],(1-test_size)*100,test_size*100,
                                                     accuracy,
                                                     f1))
