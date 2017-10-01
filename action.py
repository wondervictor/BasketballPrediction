# -*- coding: utf-8 -*-

"""
Main
"""
import argparse
import train_dnn
import data_process as dp
import models.team as team
import train_svm as svm
import train_bayes as bayes
import numpy as np


def get_team_representations(type):
    """
    Get Team Data
    :param type: representation type
    :type type: str
    :return: team data
    :rtype: dict()
    """
    team_data = dp.TeamData()
    team_data.process()
    team_raw_data = team_data.get_team()
    return team.team_representations(team_raw_data, type)


# DNN Model
# Train DNN Model
def train_with_dnn(opt):
    team_data = get_team_representations(opt.team_data_type)
    train_dnn.train_dnn_batch(opt.epoch, team_data, opt)


# Test DNN Model
def test_with_dnn(opt):
    team_data = get_team_representations(opt.team_data_type)
    train_dnn.test(team_data, opt)


# Predict DNN Model
def predict_dnn(opt):
    team_data = get_team_representations(opt.team_data_type)
    train_dnn.predict_result(team_data, opt)


# Train SVM Model
def train_svm(opt):
    team_data = get_team_representations(opt.team_data_type)
    svm.train(team_data, opt)


def predict_svm(opt):
    pass


def test_svm(opt):
    team_data = get_team_representations(opt.team_data_type)
    svm.test(team_data)


# Train Bayes Model
def train_bayes(opt):
    team_data = get_team_representations(opt.team_data_type)
    bayes.train(team_data, opt)


# Test Bayes Model
def predict_bayes(opt):
    team_data = get_team_representations(opt.team_data_type)
    bayes.train(team_data, opt)


def test_bayes(opt):
    team_data = get_team_representations(opt.team_data_type)
    bayes.test(team_data, opt)


# XGBoost
def train_xgboost(opt):
    import train_xgboost as xgboost
    team_data = get_team_representations(opt.team_data_type)
    xgboost.train(team_data, opt)


def test_xgboost(opt):
    import train_xgboost as xgboost
    team_data = get_team_representations(opt.team_data_type)
    xgboost.test(team_data)


def predict_xgboost(opt):
    import train_xgboost as xgboost

    team_data = get_team_representations(opt.team_data_type)
    xgboost.predict(team_data)


def __test_team__():

    team_data = dp.TeamData()
    team_data.process()
    team_raw_data = team_data.get_team()
    team.average({1:team_raw_data[1], 2:team_raw_data[2]}, 10)


