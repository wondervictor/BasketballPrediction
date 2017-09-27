# -*- coding: utf-8 -*-

"""
Main
"""
import argparse
import data_process as dp
import models.team as team
import numpy as np
import train_dnn


def get_team_representations(type):

    team_data = dp.TeamData()
    team_data.process()
    team_raw_data = team_data.get_team()
    return team.team_representations(team_raw_data, type)



def train_with_dnn(opt):
    print(opt.batch_size)
    team_data = get_team_representations(opt.team_data_type)
    train_dnn.train_dnn(opt.epoch, team_data, opt)


def test_with_dnn(opt):
    team_data = get_team_representations(opt.team_data_type)
    train_dnn.test(team_data, opt)


def predict(opt):
    team_data = get_team_representations(opt.team_data_type)
    train_dnn.predict_result(team_data, opt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='basketball game prediction')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--epoch', type=int, default=10,
                    help='epoch number')               
    parser.add_argument('--cuda', type=int, default=1,
                    help='CUDA training')
    parser.add_argument('--train', type=int, default=0,
                    help='CUDA training')
    parser.add_argument('--test', type=int, default=0,
                    help='CUDA training') 
    parser.add_argument('--predict', type=int, default=0,
                    help='predict result')                               
    parser.add_argument('--model_name', type=str, default='epoch_30_params.pkl',
                    help='model name')
    parser.add_argument('--team_data_type', type=str, default='average',
                    help='team data type')
    args = parser.parse_args()
    if args.train == 1:
        train_with_dnn(args)
    if args.test == 1:
        test_with_dnn(args)
    if args.predict == 1:
        predict(args)
