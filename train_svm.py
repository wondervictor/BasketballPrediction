# -*- coding: utf-8 -*-

import models.svm as svm
from data_process import test_data, train_data_func
import numpy as np


def train(team_raw_data):

    train_x = []
    train_y = []
    train_data = train_data_func()
    for x in train_data:
        home = x[1]
        away = x[0]
        home_vector = team_raw_data[home]
        away_vector = team_raw_data[away]

        away_state = x[2:4]
        home_state = x[4:6]

        input_vector = home_vector.tolist() + home_state + away_vector.tolist() + away_state
        #input_vector = np.array(input_vector)

        train_x.append(input_vector)
        train_y.append(x[-1])

    __svm = svm.svm_model(train_x, train_y)

    svm.save_model(__svm, 'svm_model_params.pkl')


def test(team_raw_data):
    train_data = test_data()
    svm_model = svm.load_model('svm_model_params.pkl')
    log_file = open('log/svm.log', 'w+')
    correct = 0
    wrong = 0
    for x in train_data:
        home = x[1]
        away = x[0]
        home_vector = team_raw_data[home]
        away_vector = team_raw_data[away]

        away_state = x[2:4]
        home_state = x[4:6]

        input_vector = home_vector.tolist() + home_state + away_vector.tolist() + away_state

        pred = svm.predict(svm_model, input_vector)
        line = 'SVM: Pred:%s Real: %s' % (pred[0], x[-1])
        if pred[0] == x[-1]:
            correct += 1
        else:
            wrong += 1
        print(line)
        log_file.write(line+'\n')

    print("Correct: %s Wrong: %s" % (correct, wrong))
    print("Acc=%s" % (float(correct)/float(correct+wrong)))
    log_file.close()









