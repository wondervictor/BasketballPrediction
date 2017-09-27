# -*- coding: utf-8 -*-

from data_process import test_data, train_data_func
import numpy as np
import models.boost as boost


def train(team_raw_data):

    train_x = []
    train_y = []
    train_data = train_data_func()
    __boost = boost.Boost()
    for x in train_data:
        home = x[1]
        away = x[0]
        home_vector = team_raw_data[home]
        away_vector = team_raw_data[away]

        away_state = x[2:4]
        home_state = x[4:6]

        input_vector = home_vector.tolist() + home_state + away_vector.tolist() + away_state

        train_x.append(input_vector)
        train_y.append(x[-1])

    __boost.train(train_x, train_x)
    __boost.save_model()


def test(team_raw_data):
    train_x = []
    y = []
    train_data = train_data_func()
    __boost = boost.Boost()
    for x in train_data:
        home = x[1]
        away = x[0]
        home_vector = team_raw_data[home]
        away_vector = team_raw_data[away]

        away_state = x[2:4]
        home_state = x[4:6]

        input_vector = home_vector.tolist() + home_state + away_vector.tolist() + away_state

        train_x.append(input_vector)

        y.append(x[-1])
        __boost.predict(train_x)



