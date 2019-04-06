# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
from iodata import data
from learning.learning import classification


# config
np.random.seed(0)

# reading the current directory
current_directory = os.getcwd()

if __name__ == '__main__':
    train = data.load_data(current_directory, test_data=False, correcting=True)
    test = data.load_data(current_directory, test_data=True, correcting=True)

    print('Training size: {}'.format(train.shape), ', Test size: {}'.format(test.shape))

    X_train, y_train = train, train['label']
    X_test, y_test = test, test['label']
    cls = classification(scoring='f1_macro', verbose=2, n_jobs=-1)
    cls.run_clf(X_train, y_train, X_test, y_test, test_labels=False, apply_cv=True, main_path=current_directory)
