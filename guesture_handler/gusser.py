# load all the gestures saved in ./guestures

import os
from typing import Dict, List

import numpy as np

from grid import Point
from save_load_points import load_points


def load_all_gestures() -> Dict[ str, List[Point]]:
    """ Loads all the gestures saved in ./guestures """
    guestures = {}
    for file_name in os.listdir('./guestures'):
        if file_name.endswith('.txt'):
            file_name_without_extension = file_name[:-4]
            guestures[file_name_without_extension] = load_points(file_name_without_extension)
    return guestures

# create a ml model to predict the gesture 
def predict(gesture: List[Point]) -> str:
    """ Predicts the gesture from the gesture points """
    guestures = load_all_gestures()
    score_dict = {}
    for name, points in guestures.items():
        sc = score(points, gesture)
        print(sc)
        score_dict[name] = sc
    
    print(score_dict)
    return max(score_dict, key= lambda key: score_dict[key] )

def score(p1: List[Point], p2: List[Point]) -> float:
    """ Scores the similarity between two gestures """

    # pad the vectors to be the same length
    if len(p1) > len(p2):
        p2 += [(0, 0)] * (len(p1) - len(p2))
    elif len(p2) > len(p1):
        p1 += [(0, 0)] * (len(p2) - len(p1))

    A = np.array(p1)
    B = np.array(p2)

    print("Arrya shape", A.shape, B.shape)

    # calculate distance using cosine similarity
    return np.dot(A, B.T) / (np.linalg.norm(A) * np.linalg.norm(B))
