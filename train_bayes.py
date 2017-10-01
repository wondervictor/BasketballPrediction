# -*- coding: utf-8 -*-

import models.Bayes as bayes
from data_process import train_data_func, test_data_func, tmp_load


def train(team_raw_data, opt):

    train_x = []
    train_y = []
    train_data = train_data_func()
    if opt.dataset == "all":
        test_data = test_data_func()
        train_data += test_data_func

    bayes_model = bayes.Bayes()
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

    bayes_model.train(train_x, train_y)


def test(team_raw_data, opt):
    __test_data = test_data_func()
    bayes_model = bayes.Bayes()
    bayes_model.load_model()
    log_file = open('log/svm.log', 'w+')
    correct = 0
    wrong = 0
    for x in __test_data:
        home = x[1]
        away = x[0]
        home_vector = team_raw_data[home]
        away_vector = team_raw_data[away]

        away_state = x[2:4]
        home_state = x[4:6]

        input_vector = home_vector.tolist() + home_state + away_vector.tolist() + away_state

        pred = bayes_model.predict(input_vector)
        print(pred)
        pred_id = 1
        if pred < 0.5:
            pred_id = 0
        line = 'Bayes: Pred:%s Real: %s Confidence=%s' % (pred_id, x[-1], pred)
        if pred_id == x[-1]:
            correct += 1
        else:
            wrong += 1
        print(line)
        log_file.write(line+'\n')

    print("Correct: %s Wrong: %s" % (correct, wrong))
    print("Acc=%s" % (float(correct)/float(correct+wrong)))
    log_file.close()


def predict(team_raw_data, opt):
    testing_data = tmp_load()
    output_file = open('output/predictPro.csv', 'w+')
    output_file.write('主场赢得比赛的置信度\n')

    bayes_model = bayes.Bayes()
    bayes_model.load_model()
    testing_input = []
    for x in testing_data:
        home = x[1]
        away = x[0]
        home_vector = team_raw_data[home]
        away_vector = team_raw_data[away]

        away_state = x[2:4]
        home_state = x[4:6]

        input_vector = home_vector.tolist() + home_state + away_vector.tolist() + away_state
        testing_input.append(input_vector)

    pred = bayes_model.predict(testing_input)
    for prob in pred:
        line = '%s\n' % prob
        output_file.write(line)

    output_file.close()