import random
import numpy as np
from numpy.typing import NDArray


def hammingDist(vec1, vec2):
    dist = 0
    for i in range(len(vec1)):
        if vec1[i] != vec2[i]:
            dist += 1
    return dist

def hammingDistNormalized(vec1, vec2):
    return hammingDist(vec1, vec2) / len(vec1) if len(vec1) != 0 else 1.0

def minHammingDist(vec1, vec2):
    dist1 = hammingDist(vec1, vec2)
    dist2 = len(vec1) - dist1
    return min(dist1, dist2)

def generateBitErrors(vector, num_errors):  
    error_vector = vector.copy()
    indices = random.sample(range(len(vector)), num_errors)

    for index in indices:
        error_vector[index] = 1 - error_vector[index]

    return error_vector

def generateBitVec(length) -> NDArray:
    return np.array([random.randint(0, 1) for _ in range(length)])

def pickRandomSubset(set, size):
    return list(sorted(random.sample(set, size)))