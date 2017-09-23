# -*- coding: utf-8 -*-

"""
DNN for Competition
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

current_comp_vector=8
team_vector=1024


class DNN(nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        self.input_team_home_layer = nn.Linear(current_comp_vector+team_vector, 1024)
        self.input_team_away_layer = nn.Linear(current_comp_vector+team_vector, 1024)
        self.home_team_layer = nn.Linear(1024, 512)
        self.away_team_layer = nn.Linear(1024, 512)
        self.comp_layer_1 = nn.Linear(1024, 1024)
        self.comp_layer_2 = nn.Linear(1024, 512)
        self.comp_layer_3 = nn.Linear(512, 256)
        self.out_prob = nn.Linear(256, 2)
        self.out_score = nn.Linear(256, 2)

    def forward(self, home_team_vector, away_team_vector, home_current_comp_vector, away_current_comp_vector):
        """
        
        :param home_team_vector: Home Team Representation by team members
        :type home_team_vector: Tensor [1x1024]
        :param away_team_vector: Away Team Representation by team members
        :type away_team_vector: Tensor [1x1024]
        :param home_current_comp_vector: Current Competition State Info for Home Team
        :type home_current_comp_vector: Tensor [1x8]
        :param away_current_comp_vector: Current Competition State Info for Away Team
        :type away_current_comp_vector: Tensor [1x8]
        :return: output_probability output_score
        :rtype: 
        """
        home_representation = F.leaky_relu(
            torch.cat([home_team_vector, home_current_comp_vector]),
            negative_slope=-0.1
        )

        away_representation = F.leaky_relu(
            torch.cat([away_team_vector, away_current_comp_vector]),
            negative_slope=-0.1
        )

        home_ready = F.tanh(
            self.home_team_layer(home_representation)
        )

        away_ready = F.tanh(
            self.away_team_layer(away_representation)
        )

        competition_round = F.relu(self.comp_layer_1(home_ready, away_ready))
        competition_round = F.leaky_relu(self.comp_layer_2(competition_round), negative_slope=-0.5)
        competition_round = F.leaky_relu(self.comp_layer_3(competition_round), negative_slope=-0.5)

        output_prob = F.softmax(self.out_prob(competition_round))
        output_score = self.out_score(competition_round)

        return output_prob, output_score


def train_dnn():
    """
    train dnn here
    :return: 
    :rtype: 
    """
    pass











