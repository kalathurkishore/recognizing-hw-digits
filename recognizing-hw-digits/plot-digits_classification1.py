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

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np

digits = datasets.load_digits()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
test_size = 0.15
validation_size = 0.15

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=test_size + validation_size, shuffle=False)

X_test, X_validate, y_test, y_validate = train_test_split(
    X_test, y_test, test_size=validation_size/(test_size+validation_size), shuffle=False)

gamma_val_array = [0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000]
gamma_max_acc = 0
best_gamma = 0
for gamma_i in gamma_val_array:
        clf = svm.SVC(gamma=gamma_i)
        clf.fit(X_train, y_train)
        predicted_train_set = clf.predict(X_train)
        predicted_validate_set = clf.predict(X_validate)
        predicted_test_set = clf.predict(X_test)
        print("Accuracy of ",gamma_i,"          on train set is",metrics.accuracy_score(predicted_train_set,y_train),"  validate set is",metrics.accuracy_score(predicted_validate_set,y_validate),"            on test set is ",metrics.accuracy_score(predicted_test_set,y_test))
        if metrics.accuracy_score(predicted_validate_set,y_validate) > gamma_max_acc:
                best_gamma = gamma_i
                gamma_max_acc=metrics.accuracy_score(predicted_validate_set,y_validate)
clf = svm.SVC(gamma=best_gamma)
clf.fit(X_train,y_train)
predicted_test_set = clf.predict(X_test)
predicted_train_set = clf.predict(X_train)
#predicted_valid = clf.predict(X_valid)
#valid_accuracy = metrics.accuracy_score(predicted_valid,y_train)
train_accuracy = metrics.accuracy_score(predicted_train_set,y_train)
test_accuracy = metrics.accuracy_score(predicted_test_set,y_test)
print("Best gamma is:",best_gamma," with validation accuracy: ",gamma_max_acc,"train accuracy: ",train_accuracy,"test_accuracy: ",test_accuracy)
