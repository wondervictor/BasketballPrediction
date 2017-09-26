# -*- coding: utf-8 -*-

"""
DNN for Competition
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from data_preprocess import MatchData
from team import Team
import os
import numpy as np
import torch.optim as optimizer
import random

CURRENT_COMP_VECTOR_SIZE = 2
TEAM_VECTOR_SIZE = 21


class DNN(nn.Module):

    def __init__(self):
        super(DNN, self).__init__()
        self.input_team_home_layer = nn.Linear(CURRENT_COMP_VECTOR_SIZE+TEAM_VECTOR_SIZE, 128)
        self.input_team_away_layer = nn.Linear(CURRENT_COMP_VECTOR_SIZE+TEAM_VECTOR_SIZE, 128)
        self.home_team_layer = nn.Linear(128, 256)
        self.away_team_layer = nn.Linear(128, 256)
        self.comp_layer_1 = nn.Linear(512, 512)
        self.comp_layer_2 = nn.Linear(512, 256)
        self.comp_layer_3 = nn.Linear(256, 256)
        self.comp_layer_4 = nn.Linear(256, 128)

        self.out_prob = nn.Linear(128, 2)

        self.out_score = nn.Linear(128, 2)

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
            self.input_team_home_layer(torch.cat([home_team_vector, home_current_comp_vector], dim=1)),
            negative_slope=-0.1
        )

        away_representation = F.leaky_relu(
            self.input_team_away_layer(torch.cat([away_team_vector, away_current_comp_vector], dim=1)),
            negative_slope=-0.1
        )

        home_ready = F.tanh(
            self.home_team_layer(home_representation)
        )

        away_ready = F.tanh(
            self.away_team_layer(away_representation)
        )

        competition_round = F.relu(self.comp_layer_1(torch.cat([home_ready, away_ready], dim=1)))
        competition_round = F.leaky_relu(self.comp_layer_2(competition_round), negative_slope=-0.5)
        competition_round = F.leaky_relu(self.comp_layer_3(competition_round), negative_slope=-0.5)
        competition_round = F.leaky_relu(self.comp_layer_4(competition_round), negative_slope=-0.5)

        output_prob = F.softmax(self.out_prob(competition_round))
        output_score = self.out_score(competition_round)

        return output_prob, output_score


def save_model(net, name):
    path = 'model/'
    torch.save(net, path+name)


def load_model(name):
    path = 'model/'
    return torch.load(path+name)


def predict(model_name, home_vector, away_vector, home_state, away_state, opt):

    net = load_model(model_name)
    if opt.cuda == 1:
        home_current_state = Variable(torch.FloatTensor(home_state).cuda())
        away_current_state = Variable(torch.FloatTensor(away_state).cuda())

        away_vector = Variable(torch.FloatTensor(away_vector).cuda())
        home_vector = Variable(torch.FloatTensor(home_vector).cuda())
    else:
        home_current_state = Variable(torch.FloatTensor(home_state))
        away_current_state = Variable(torch.FloatTensor(away_state))

        away_vector = Variable(torch.FloatTensor(away_vector))
        home_vector = Variable(torch.FloatTensor(home_vector))

    home_current_state = home_current_state.unsqueeze(0)
    away_current_state = away_current_state.unsqueeze(0)
    home_vector = home_vector.unsqueeze(0)
    away_vector = away_vector.unsqueeze(0)

    prob, _ = net(
        home_current_comp_vector=home_current_state,
        home_team_vector=home_vector,
        away_current_comp_vector=away_current_state,
        away_team_vector=away_vector,
    )
    prob = prob.squeeze(0)
    return prob


def train_dnn_batch(epoches, team_data, opt):
    """
    train mini batch dnn here
    :return: 
    :rtype: 
    """

    batch_size = opt.batch_size
    dnn = DNN()
    if opt.cuda == 1:
        dnn.cuda()
    data_provider = MatchData(1000)
    dnn_optimizer = optimizer.Adamax(dnn.parameters(), lr=0.001)
    prob_criterion = torch.nn.CrossEntropyLoss()
    score_criterion = torch.nn.MSELoss()

    print("Starting to train with DNN")
    for epoch in range(epoches):
        data_provider.roll_data()
        train_data = data_provider.get_train_data()

        for i in range(0, len(train_data), batch_size):
            batch_home_current_state = Variable(torch.zeros((batch_size, CURRENT_COMP_VECTOR_SIZE)))
            batch_away_current_state = Variable(torch.zeros((batch_size, CURRENT_COMP_VECTOR_SIZE)))

            batch_home_vector = Variable(torch.zeros((batch_size, TEAM_VECTOR_SIZE)))
            batch_away_vector = Variable(torch.zeros((batch_size, TEAM_VECTOR_SIZE)))

            batch_score = Variable(torch.zeros((batch_size, 2)))
            batch_result = []

            for p in range(batch_size):
                away_id = train_data[i+p][0]
                home_id = train_data[i+p][1]

                away_current_state = train_data[i+p][2:4]
                home_current_state = train_data[i+p][4:6]
                score = [train_data[i+p][7], train_data[i+p][6]]
                away_vector = team_data[away_id]
                home_vector = team_data[home_id]
                result = train_data[i+p][8]

                batch_home_current_state[p] = Variable(torch.FloatTensor(home_current_state))
                batch_away_current_state[p] = Variable(torch.FloatTensor(away_current_state))
                batch_away_vector[p] = Variable(torch.FloatTensor(away_vector))
                batch_home_vector[p] = Variable(torch.FloatTensor(home_vector))
                batch_score[p] = Variable(torch.FloatTensor(score))
                batch_result.append(result)
            batch_result = Variable(torch.LongTensor(batch_result))
            if opt.cuda == 1:
                batch_home_current_state = batch_home_current_state.cuda()
                batch_away_current_state = batch_home_current_state.cuda()
                batch_home_vector = batch_home_current_state.cuda()
                batch_away_vector = batch_home_current_state.cuda()
                batch_result = batch_result.cuda()
                batch_score = batch_score.cuda()

            output_prob, output_score = dnn.forward(
                home_current_comp_vector=batch_home_current_state,
                home_team_vector=batch_home_vector,
                away_current_comp_vector=batch_away_current_state,
                away_team_vector=batch_away_vector,
            )
            loss = prob_criterion(output_prob, batch_result)
            #loss += 0.001*score_criterion(output_score, batch_score)

            dnn_optimizer.zero_grad()
            loss.backward()
            dnn_optimizer.step()
            if i % 10 == 0:
                print("Batch: %s Loss: %s" % (i+1, loss.data[0]))
        save_model(dnn, 'epoch_%d_params.pkl' % epoch)


def train_data():
    with open('data/train.csv', 'r') as openfile:
        lines = openfile.readlines()

    data = []
    for line in lines[1:]:
        data.append(map(int, line.split(',')))

    random.shuffle(data)
    return data


def test_data():

    with open('data/test.csv', 'r') as openfile:
        lines = openfile.readlines()

    data = []
    for line in lines[1:]:
        data.append(map(int, line.split(',')))
    random.shuffle(data)
    return data


def train_dnn(epoches, team_data, opt):
    """
    train dnn here
    :return: 
    :rtype: 
    """

    dnn = DNN()
    if opt.cuda == 1:
        dnn.cuda()
    #data_provider = MatchData(1000)
    dnn_optimizer = optimizer.Adamax(dnn.parameters(), lr=0.001)
    prob_criterion = torch.nn.CrossEntropyLoss()
    score_criterion = torch.nn.MSELoss()

    print("Starting to train with DNN")
    for epoch in range(epoches):
        #data_provider.roll_data()
        #train_data = data_provider.get_train_data()
        train_data = train_data()

        for i in range(len(train_data)):
            #     Competition: [(Away, Home, Away_Ago_Win, Away_Ago_Lose, Home_Ago_Win, Home_Ago_Lose, Away_Score, Home_Score, Home_Win)]
            away_id = train_data[i][0]
            home_id = train_data[i][1]

            away_current_state = train_data[i][2:4]
            home_current_state = train_data[i][4:6]
            score = [train_data[i][7], train_data[i][6]]
            away_vector = team_data[away_id]
            home_vector = team_data[home_id]
            result = [train_data[i][8]]


            if opt.cuda == 1:
                home_current_state = Variable(torch.FloatTensor(home_current_state).cuda())
                away_current_state = Variable(torch.FloatTensor(away_current_state).cuda())

                away_vector = Variable(torch.FloatTensor(away_vector).cuda())
                home_vector = Variable(torch.FloatTensor(home_vector).cuda())
                prob = Variable(torch.LongTensor(result).cuda())
                score = Variable(torch.FloatTensor(score).cuda())
            else:    
                home_current_state = Variable(torch.FloatTensor(home_current_state))
                away_current_state = Variable(torch.FloatTensor(away_current_state))

                away_vector = Variable(torch.FloatTensor(away_vector))
                home_vector = Variable(torch.FloatTensor(home_vector))
                prob = Variable(torch.LongTensor(result))

                score = Variable(torch.FloatTensor(score))

            home_current_state = home_current_state.unsqueeze(0)
            away_current_state = away_current_state.unsqueeze(0)
            home_vector = home_vector.unsqueeze(0)
            away_vector = away_vector.unsqueeze(0)

            output_prob, output_score = dnn.forward(
                home_current_comp_vector=home_current_state,
                home_team_vector=home_vector,
                away_current_comp_vector=away_current_state,
                away_team_vector=away_vector,
            )
            loss = prob_criterion(output_prob, prob)
            #loss += 0.001*score_criterion(output_score, score)

            dnn_optimizer.zero_grad()
            loss.backward()
            dnn_optimizer.step()
            if i % 10 == 0:
                print("Sample: %s Loss: %s" % (i+1, loss.data[0]))

        save_model(dnn, 'epoch_%d_params.pkl' % epoch)


def test(team_data, opt):
    data_provider = MatchData(1000)
    data_provider.roll_data()

    #testing_data = data_provider.get_test_data()
    testing_data = test_data()
    log_file = open('testing.log', 'w+')

    correct = 0
    wrong = 0

    for i in range(len(testing_data)):

        away_id = testing_data[i][0]
        home_id = testing_data[i][1]

        away_current_state = testing_data[i][2:4]
        home_current_state = testing_data[i][4:6]
        score = [testing_data[i][7], testing_data[i][6]]
        away_vector = team_data[away_id]
        home_vector = team_data[home_id]
        result = [testing_data[i][8]]

        prob = predict(
            opt.model_name,
            home_state=home_current_state,
            home_vector=home_vector,
            away_state=away_current_state,
            away_vector=away_vector,
            opt=opt
        )

        pred_win = np.argmax(prob.data.cpu().numpy())

        if pred_win == result:
            correct += 1
            line = 'Test: %s Correct! Confidence=%s' % (i, prob.data[pred_win])
        else:
            wrong += 1
            line = 'Test: %s Wrong! Confidence=%s' % (i, prob.data.cpu().numpy().tolist())

        print(line)
        log_file.write(line+'\n')
    log_file.close()

    print("Wrong: %s Correct: %s" % (wrong, correct))



# class network():
    
#     def __init__(self):
#         pass
    
#     def set_input(self):






















