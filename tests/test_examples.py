import math

def test_equality():
	assert 10 == 10

def test_sqrt():
	num = 121
	assert math.sqrt(num) == 11

def test_square():
	num = 11
	assert 11*11 == 121

import math
import recognizing-hw-digits.util as util
import os
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

def test_model_writing():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    rescale_factor=1
    (test_size, valid_size) =(0.15, 0.15)

    X_train, X_test,X_valid,y_train,y_test,y_valid=util.createsplit(data,digits.target,test_size,valid_size)

    gamma = 0.001
    classifier = svm.SVC(gamma=gamma)
    output_folder="./mymodel_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, valid_size, rescale_factor, gamma
            )
    output_model_file=os.path.join(output_folder,"model.joblib")

    util.run_classification_experiment(classifier, X_train, y_train, X_valid, y_valid, gamma, output_model_file)

    assert os.path.isfile(output_model_file)

def test_small_data_overfit_checking():
    digits = datasets.load_digits()
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    targets = digits.target[:int(0.4*len(data))]
    data = data[:int(0.4*len(data))]

    x_train, x_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.2, shuffle=False)

    x_val, x_test, y_val, y_test = train_test_split(
        x_test,y_test, test_size=0.5, shuffle=False)

    gamma = 0.001
    classifier = svm.SVC(gamma=gamma)
    test_size=0.4
    valid_size=0.4
    rescale_factor=1
    output_folder="./mymodel_{}_val_{}_rescale_{}_gamma_{}".format(
                test_size, valid_size, rescale_factor, gamma
            )
    output_model_file=os.path.join(output_folder,"model.joblib")
    train_metrics = util.run_classification_experiment(classifier, x_train, y_train, x_val, y_val, gamma, output_model_file)

    assert train_metrics['acc']  > 0.8
    assert train_metrics['f1'] > 0.7

