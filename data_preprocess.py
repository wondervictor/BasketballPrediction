#-*- coding: utf-8 -*-

import random
import numpy as np

__all__ = ['MatchData']


def load_competitions():
    """
    Load Comepetions
    Competition: [(Away:int, Home:int, Away_Ago:[int], Home_Ago:[int], rate:[int])]
    """

    data = []
    with open('data/matchDataTrain.csv') as open_file:
        lines = open_file.readlines()

    def get_record(rec_str):
        rec_str = rec_str.replace('胜', ',')
        rec_str = rec_str.replace('负', ',')
        records = map(int, rec_str.split(',')[:-1])

        return records

    for line in lines[1:]:
        elements = line.split(',')
        away = int(elements[0])
        home = int(elements[1])

        away_ago = get_record(elements[2])
        home_ago = get_record(elements[3])

        score = map(int, elements[4].split(':'))

        data.append([away, home, away_ago, home_ago, score])

    return data


class MatchData(object):

    def __init__(self, testing_size):
        self.data = load_competitions()
        random.shuffle(self.data)
        self.current_index = 0
        self.testing_size = testing_size

    def roll_data(self):
        """
        :description: cross-validation shuffle data when the index > size 
        """
        if self.current_index > len(self.data):
            self.current_index = 0
            random.shuffle(self.data)
        self.testing_data = self.data[self.current_index:self.current_index+self.testing_size]
        self.training_data = self.data[:self.current_index]+self.data[self.current_index+self.testing_size:]
        self.current_index += self.testing_size

    def get_train_data(self):
        self.roll_data()
        return self.training_data

    def get_test_data(self):
        return self.testing_data


def test_load_competitions():

    data = load_competitions()
    print(data[1:10])



if __name__ == '__main__':

    test_load_competitions()











