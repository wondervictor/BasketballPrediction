# -*- coding: utf-8 -*-

"""
Team Model
"""
from sklearn.decomposition import PCA
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

        weights = [0.14, 0.14, 0.14, 0.14, 0.14, 0.1, 0.1, 0.1]

        def get_top_k(k, team):
            priority_queue = Queue.PriorityQueue()
            for i in range(len(team)):
                member = team[i]
                priority_queue.put((-(member[0]+member[1] + 0.01 * i), member))
            priority_team_data = []
            for i in range(k):
                s = priority_queue.get()
                priority_team_data.append(np.array(s[-1]))
            return np.array(priority_team_data)

        #__pca = PCA(n_components=18, whiten=True, copy=False)
        for key in team_raw_data.keys():
            team = team_raw_data[key]
            team = get_top_k(8, team)
            # for member in team:
            #     member = __pca.fit_transform(member)
            #     print(member.shape)
            # __pca.fit(team)
            # team = __pca.transform(team)
            team_vector = team[0]
            for i in range(1, 8):
                team_vector += team[i]*weights[i]
            team_data[key] = team_vector

        return team_data

    else:
        print("Not Implemented!")

    print("Team Data Prepared Finished!")
    return team_data
