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
            team_vector = team[0]*weights[0]
            for i in range(1, 8):
                team_vector += team[i]*weights[i]
            team_data[key] = team_vector
        return team_data
    elif type == 'reduce':

        """
            Member: [投篮命中率, 三分命中率, 罚球命中率, 篮板总数, 前场篮板, 后场篮板, 助攻, 抢断, 盖帽, 失误, 犯规, 得分]
                    [0出场次数,1首发次数,2上场时间,3投篮命中率,4投篮命中次数,5投篮出手次数,6三分命中率,7三分命中次数,8三分出手次数,9罚球命中率,
                    10罚球命中次数,11罚球出手次数,12篮板总数,13前场篮板,14后场篮板,15助攻,16抢断,17盖帽,18失误,19犯规,20得分]
        """

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

        def retrieve_essentials(member):
            member[18] = -member[18]
            member[19] = -member[19]
            return [member[3], member[6],member[9]] + member[12:]

        for key in team_raw_data.keys():

            team = team_raw_data[key]
            team = get_top_k(8, team)
            team_vector = retrieve_essentials(team[0]) * weights[0]
            for i in range(1, 8):
                team_vector += retrieve_essentials(team[i]) * weights[i]
            team_data[key] = team_vector
        return team_data
    else:
        print("Not Implemented!")

    print("Team Data Prepared Finished!")
    return team_data
