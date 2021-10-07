from skimage.transform import rescale
import numpy as np
import os
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from joblib import dump, load

def preprocess(images,rescale_factor):
   resized_images = []
   for d in images:
      resized_images.append(rescale(d,rescale_factor,anti_aliasing=False))
   return resized_images

def createsplit(data,targets,test_size,valid_size):
  X_train, X_test_valid, y_train, y_test_valid = train_test_split(data, targets, test_size=test_size + valid_size, shuffle=False)
  X_test, X_valid, y_test, y_valid = train_test_split(X_test_valid,y_test_valid,test_size=valid_size / (test_size + valid_size),shuffle=False)
  return X_train, X_test,X_valid,y_train,y_test,y_valid

def test(clf,X,y):
  predicted = clf.predict(X)
  acc = metrics.accuracy_score(y_pred=predicted, y_true=y)
  f1 = metrics.f1_score(y_pred=predicted, y_true=y, average="macro")
  return {'acc':acc,'f1':f1}

def run_classification_experiment(clf,X_train,y_train,X_valid,y_valid,gamma,output_model_file,skip_dummy=True):
    clf.fit(X_train, y_train)
    metrics_value=test(clf,X_valid,y_valid)
    if skip_dummy and metrics_value["acc"]<0.11:
        print("SKipping for {}".format(gamma))
        return None
    output_folder = os.path.dirname(output_model_file)
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    dump(clf, output_model_file)
    return metrics_value