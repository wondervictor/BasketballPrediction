# -*- coding: utf-8 -*-

"""
Team Model
"""
from sklearn.decomposition import pca
import numpy as np
import Queue


def team_representations(team_raw_data, type):
    """
    Process Team Data to Retrieve its Representations
    :return Tensor [1x21]
    """
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
    elif type == "rank_8":
        for key in team_raw_data.keys():
            team = team_raw_data[key]
            team_vector = team[0]
            for i in range(1, 8):
                team_vector += team[i]
            team_vector = team_vector/8
            team_data[key] = team_vector
    elif type == "pca":

        weights = [0.16, 0.16, 0.16, 0.14, 0.12, 0.10, 0.80, 0.80]

        def get_top_k(k, team):
            priority_queue = Queue.PriorityQueue()
            for member in team:
                priority_queue.put(-member[1], member)
            priority_team_data = []
            for i in range(k):
                priority_team_data.append(priority_queue.get())
            return team_vector

        __pca = pca.PCA(n_components=18, whiten=True, copy=True)
        for key in team_raw_data.keys():
            team = team_raw_data[key]
            team = get_top_k(8, team)
            team = __pca.fit_transform(team)
            team_vector = team[0]
            for i in range(1, 8):
                team_vector += team[i]*weights[i]
            team_data[key] = team_vector
            print(team_vector)
        return team_data

    else:
        print("Not Implemented!")

    print("Team Data Prepared Finished!")
    return team_data
