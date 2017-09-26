# -*- coding: utf-8 -*-

"""
Main
"""
import argparse
import data_preprocess as dp
import dnn as dnn

import numpy as np


def team_representations(type):
    """
    Process Team Data to Retrieve its Representations
    :return Tensor [1x21]
    """

    team_data = dp.TeamData()
    team_data.process()
    team_raw_data = team_data.get_team()

    team_data = dict()
    # Average
    if type == "average":
        for key in team_raw_data.keys():
            team = team_raw_data[key]
            team_vector = team[0]
            for i in range(1, len(team)):
                team_vector += team[i]

            team_vector = team_vector/len(team)
            team_data[key] = team_vector
    else:
        print("Not Implemented!")

    print("Team Data Prepared Finished!")
    return team_data


def train_with_dnn(opt):
    print(opt.batch_size)
    team_data = team_representations('average')
    dnn.train_dnn_batch(50,team_data, opt)


def test_with_dnn(opt):
    team_data = team_representations('average')
    dnn.test(team_data, opt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='basketball game prediction')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--cuda', type=int, default=1,
                    help='CUDA training')
    parser.add_argument('--train', type=int, default=0,
                    help='CUDA training')
    parser.add_argument('--test', type=int, default=0,
                    help='CUDA training')                               
    args = parser.parse_args()
    if args.train == 1:
        train_with_dnn(args)
    if args.test == 1:
        test_with_dnn(args)
