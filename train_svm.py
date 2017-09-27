# -*- coding: utf-8 -*-

import models.svm as svm
from data_process import test_data, train_data_func


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

        input_vector = home_vector+home_state+away_vector+away_state

        train_x.append(input_vector)
        train_y.append(x[-1])
        print(input_vector)
        print(train_y)








