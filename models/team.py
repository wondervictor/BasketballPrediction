# -*- coding: utf-8 -*-

"""
Team Model
"""


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
            for i in range(1, 9):
                team_vector += team[i]
            team_vector = team_vector/8
            team_data[key] = team_vector
    else:
        print("Not Implemented!")

    print("Team Data Prepared Finished!")
    return team_data
