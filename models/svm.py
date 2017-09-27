# -*- coding: utf-8 -*-

"""
SVM for Competition
"""

from sklearn.svm import *
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler

import pickle


def svm_model(train_x, train_y):

    svm = SVR(
        max_iter=1000000,
        kernel='rbf',
        C=1.0,
        gamma=20.0,
    )

    svm.fit(train_x, train_y)
    return svm


def save_model(svm, filename):

    joblib.dump(svm, 'model_params/'+filename)


def load_model(filename):
    svm = joblib.load('model_params/'+filename)
    return svm


def predict(svm, x):
    return svm.predict(x)
