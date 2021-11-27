'''import math

def test_equality():
	assert 10 == 10

def test_sqrt():
	num = 121
	assert math.sqrt(num) == 11

def test_square():
	num = 11
	assert 11*11 == 121
'''
'''
import sys 
import os
import warnings
from sklearn.utils.random import sample_without_replacement
import math

warnings.filterwarnings("ignore")

base_path = '/home/kishore/hwdigits/recognizing-hw-digits/recognizing-hw-digits'        
sys.path.append(base_path)
from plot_digits import classification_of_data
from sklearn import datasets

def test_create_split_bonus():
	digits = datasets.load_digits()
	n_samples = len(digits.images)
	val_ratio = 0.2
	test_ratio = 0.1
	train_ratio = 1 - val_ratio - test_ratio

	train_sample = math.ceil(n_samples *train_ratio)
	test_sample = math.ceil(n_samples *test_ratio)
	val_sample = math.ceil(n_samples *val_ratio)

	actual_train ,actual_test , actual_valid ,_ ,_ ,_= create_splits(digits.images, digits.target, test_ratio, val_ratio)    
    
	total = len(actual_train) + len(actual_test) + len(actual_valid) 

	assert train_sample == len(actual_train)
	assert test_sample == len(actual_test)
	assert val_sample == len(actual_valid)
	assert n_samples == total


def test_create_split_1():
	digits = datasets.load_digits()
	n_samples = 100
	val_ratio = 0.7
	test_ratio = 0.2
	train_ratio = 1 - val_ratio - test_ratio

	train_sample = int(n_samples *train_ratio)
	test_sample = int(n_samples *test_ratio)
	val_sample = int(n_samples *val_ratio)

	actual_train ,actual_test , actual_valid ,_ ,_ ,_= create_splits(digits.images[:n_samples ], digits.target[:n_samples ], test_ratio, val_ratio)    
    
	total = len(actual_train) + len(actual_test) + len(actual_valid) 

	assert train_sample == len(actual_train)
	assert test_sample == len(actual_test)
	assert val_sample == len(actual_valid)
	assert n_samples == total

def test_create_split_2():
	digits = datasets.load_digits()
	n_samples = 9
	val_ratio = 0.7
	test_ratio = 0.2
	train_ratio = 1 - val_ratio - test_ratio

	train_sample = math.ceil(n_samples *train_ratio)
	test_sample = math.ceil(n_samples *test_ratio)
	val_sample = math.ceil(n_samples *val_ratio)

	actual_train ,actual_test , actual_valid = create_splits(digits.images[:n_samples ], digits.target[:n_samples ], test_ratio, val_ratio,case = True)    
    
	total = actual_train +actual_test + actual_valid 
'''
import sys 
import os
import warnings
#from sklearn.utils.random import sample_without_replacement
import math
from joblib import dump, load

import matplotlib.pyplot as plt
import numpy as np

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,accuracy_score
from tabulate import tabulate
from sklearn import tree
import pickle
import statistics


warnings.filterwarnings("ignore")

svm_clf = load('SVM.joblib')
dt_clf = load('DT.joblib')

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
target = digits.target

clf = load('SVM.joblib')

def test_digit_correct_0():
	count = 0
	while(1):
		if target[count] == 0:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 0)

def test_digit_correct_1():
	count = 0
	while(1):
		if target[count] == 1:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 1)


def test_digit_correct_2():
	count = 0
	while(1):
		if target[count] == 2:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 2)

def test_digit_correct_3():
	count = 0
	while(1):
		if target[count] == 3:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 3)

def test_digit_correct_4():
	count = 0
	while(1):
		if target[count] == 4:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 4)


def test_digit_correct_5():
	count = 0
	while(1):
		if target[count] == 5:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 5)


def test_digit_correct_6():
	count = 0
	while(1):
		if target[count] == 6:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 6)


def test_digit_correct_7():
	count = 0
	while(1):
		if target[count] == 7:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 7)


def test_digit_correct_8():
	count = 0
	while(1):
		if target[count] == 8:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 8)

def test_digit_correct_9():
	count = 0
	while(1):
		if target[count] == 9:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 9)


clf = load('DT.joblib')

def test_digit_dt_correct_0():
	count = 0
	while(1):
		if target[count] == 0:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 0)

def test_digit_dt_correct_1():
	count = 0
	while(1):
		if target[count] == 1:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 1)


def test_digit_dt_correct_2():
	count = 0
	while(1):
		if target[count] == 2:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 2)

def test_digit_dt_correct_3():
	count = 0
	while(1):
		if target[count] == 3:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 3)

def test_digit_dt_correct_4():
	count = 0
	while(1):
		if target[count] == 4:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 4)


def test_digit_dt_correct_5():
	count = 0
	while(1):
		if target[count] == 5:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 5)


def test_digit_dt_correct_6():
	count = 0
	while(1):
		if target[count] == 6:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 6)


def test_digit_dt_correct_7():
	count = 0
	while(1):
		if target[count] == 7:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 7)


def test_digit_dt_correct_8():
	count = 0
	while(1):
		if target[count] == 8:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 8)

def test_digit_dt_correct_9():
	count = 0
	while(1):
		if target[count] == 9:
			image = np.array(data[count]).reshape(1,-1)
			predicted = clf.predict(image )
			break 
		count += 1

	assert(predicted[0] == 9)
