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


X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.3, shuffle=False)

clf = svm.SVC(gamma=0.0000001)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print("Accuracy for gamma 0.0000001 is ", metrics.accuracy_score(predicted,y_test))

clf = svm.SVC(gamma=0.000001)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print("Accuracy for gamma 0.000001 is ", metrics.accuracy_score(predicted,y_test))

clf = svm.SVC(gamma=0.00001)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print("Accuracy for gamma 0.00001 is ", metrics.accuracy_score(predicted,y_test))

clf = svm.SVC(gamma=0.0001)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print("Accuracy for gamma 0.0001 is ", metrics.accuracy_score(predicted,y_test))

clf = svm.SVC(gamma=0.001)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print("Accuracy for gamma 0.001 is ", metrics.accuracy_score(predicted,y_test))

clf = svm.SVC(gamma=0.01)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print("Accuracy for gamma 0.01 is ", metrics.accuracy_score(predicted,y_test))

clf = svm.SVC(gamma=0.1)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print("Accuracy for gamma 0.1 is ", metrics.accuracy_score(predicted,y_test))

clf = svm.SVC(gamma=1)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print("Accuracy for gamma 1 is ", metrics.accuracy_score(predicted,y_test))

clf = svm.SVC(gamma=10)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print("Accuracy for gamma 10 is ", metrics.accuracy_score(predicted,y_test))

clf = svm.SVC(gamma=100)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print("Accuracy for gamma 100 is ", metrics.accuracy_score(predicted,y_test))

clf = svm.SVC(gamma=1000)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print("Accuracy for gamma 1000 is ", metrics.accuracy_score(predicted,y_test))

clf = svm.SVC(gamma=10000)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print("Accuracy for gamma 10000 is ", metrics.accuracy_score(predicted,y_test))

clf = svm.SVC(gamma=100000)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
print("Accuracy for gamma 100000 is ", metrics.accuracy_score(predicted,y_test))
