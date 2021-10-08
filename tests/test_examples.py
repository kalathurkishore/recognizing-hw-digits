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


def test_model_writing():
	digits = datasets.load_digits()
	metric , model_path = classification_of_data(digits)
	final_file = '../recognizing-hw-digits/' + model_path[1:] 
	assert os.path.isdir(final_file)

def test_small_data_overfit_checking():
	digits = datasets.load_digits()
	metrics , _ = classification_of_data(digits,isTrain=True)
	assert metrics['acc'] > 0.7
	assert metrics['f1'] > 0.7
