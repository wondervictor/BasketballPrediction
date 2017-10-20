# -*- coding: utf-8 -*-

"""
Team Model
"""
from sklearn.decomposition import PCA
import numpy as np
import Queue
from pprint import pprint


def member_process(member):
    """

    :param member: 
    :type member: 
    :return:[0上场时间,1投篮命中率,2投篮命中次数,3投篮出手次数,4三分命中率,5三分命中次数,6三分出手次数,
    7罚球命中率,8罚球命中次数,9罚球出手次数,10篮板总数,11前场篮板,12后场篮板,13助攻,14抢断,15盖帽,16失误,17犯规,18得分] 
    :rtype: 
    """

    member[3] = member[3]/100
    member[6] = member[6]/100
    member[9] = member[9]/100

    member[18] = -member[18]
    member[19] = -member[19]

    return member[2:]


def get_top_k(k, team):
    """
    Return the Top-K team members of a Team
    :param k: Numbers 
    :type k: int
    :param team: Team Tensor 
    :type team: 
    :return: the top k members
    :rtype: 
    """

    priority_queue = Queue.PriorityQueue()
    for i in range(len(team)):
        member = team[i]
        priority_queue.put((-(member[20] + 0.001 * i), member))
    priority_team_data = []
    for i in range(k):
        s = priority_queue.get()
        priority_team_data.append(s[-1])
    return priority_team_data


def average(team_raw_data, topk=-1):
    """
    Return the average vector of all memebers
    """

    team_data = dict()

    for key in team_raw_data.keys():
        team = team_raw_data[key]
        if topk != -1:
            team = get_top_k(topk, team)

        team_vector = member_process(team[0])
        for i in range(1, len(team)):
            team_vector += member_process(team[i])
        team_data[key] = team_vector
    return team_data


def extract_essentials(member):

    """
    :return: [投篮命中率,投篮次数, 3*三分命中率,三分次数, 罚球命中率, 篮板总数, 前场篮板, 后场篮板, 助攻, 抢断, 盖帽, 失误, 犯规, 得分]
    """
    member = member_process(member)
    new_member = [member[1], 3*member[3], member[4], member[6], member[7]] + member[10:18].tolist() + [10*member[18]]

    return np.array(new_member)


def reduce_(team_raw_data, topk=-1):
    """
    Reduce the features to 14 dimension
    :param team_raw_data: raw team data 
    :type team_raw_data: [1x21] Numpy Array
    """
    team_data = dict()

    for key in team_raw_data.keys():
        team = team_raw_data[key]
        if topk != -1:
            team = get_top_k(topk, team)
        team_vector = extract_essentials(team[0])
        for i in range(1, len(team)):
            team_vector += extract_essentials(team[i])
        team_data[key] = team_vector
    return team_data


def team_representations(team_raw_data, type, topk=-1):

    if type == "average":
        return average(team_raw_data, 8)
    elif type == "reduce":
        return reduce_(team_raw_data, 8)
    else:
        print("Not Implemented!")
        exit(-1)




