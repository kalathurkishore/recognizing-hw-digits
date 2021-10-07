import recognizing-hw-digits.utils as utils
import recognizing-hw-digits.plot-digits_classification1 as plot
from sklearn import datasets




def test_model_writing():

   1. create some data
   2. run_classification_experiment(data, expeted-model-file)
   assert os.path.isfile(expected-model-file)


def test_small_data_overfit_checking():
   1. create a small amount of data / (digits / subsampling)
   2. train_metrics = run_classification_experiment(train=train, valid=train)
   assert train_metrics['acc']  > 0.7
   assert train_metrics['f1'] > 0.7

