# -*- coding: utf-8 -*-

"""
SVM for Competition
"""

from sklearn.svm import *
import pickle


def svm_model(train_x, train_y):

    svm = SVC(
        kernel='rbf',
        C=1.0,
        max_iter=10,
        gamma=20.0
    )

    svm.fit(train_x, train_y)


def save_model(svm, filename):

    with open('model_params/%s'%filename, 'wb') as f:
        pickle.dump(svm, f)


def load_model(filename):
    with open('model_params/%s'%filename, 'rb') as f:
        svm = pickle.load(f)
    return svm


def predict(svm, x):
    return svm.predict(x)
