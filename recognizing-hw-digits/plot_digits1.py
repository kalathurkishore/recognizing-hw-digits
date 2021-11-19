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

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from tabulate import tabulate
import pickle

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

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))



# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.1, shuffle=False)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, shuffle=False)


gamma = [10**i for i in range(-7,7)]
data = []

model_lst = []
model_name_lst = []


for gm in gamma:

    clf = svm.SVC(gamma=gm)
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_val)
    f1 = round(f1_score(y_val, predicted, average='weighted'),2)
    acc_val = round(accuracy_score(y_val , predicted),2)



    if(acc_val>0.25):
        print("Storing metrics for gamma " ,gm)
        model = [clf ,f1,acc_val]
        model_lst.append(model)

        data.append([gm,f1,acc_val])

        filename = './models/model_'+str(gm)+'.sav'
        model_name_lst.append(filename)
        pickle.dump(model, open(filename, 'wb'))
        print("Saving model for gamma " , gm)

    else:
        print("Skipping for gamma",gm)

print()
print(tabulate(data, headers=["Gamma","F1-Score(weighted)", "Accuracy Val"]))
print()

max_a = 0
idx = 0


for i in range(len(data)):
    if(data[i][2] > max_a):
        idx = i
        max_a = data[i][1]


filename = model_name_lst[idx]
print()
print("Loading model corresponding to best Gamma......")
print()
clf = pickle.load(open(filename, 'rb'))[0]


predicted = clf.predict(X_test)
acc_test = round(accuracy_score(y_test , predicted),2)
print("Best Gamma Value : ",data[idx][0])
print("Test accuracy for best gamma ",acc_test)

print()
print("Done")
