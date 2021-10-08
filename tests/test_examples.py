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
import sys 
import os
import warnings
from sklearn.utils.random import sample_without_replacement

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

	assert train_sample == len(actual_train)
	assert test_sample == len(actual_test)
	assert val_sample == len(actual_valid)
	assert n_samples == total
