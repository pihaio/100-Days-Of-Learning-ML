"""
    Function:K Nearest Neighbors
    Author:pihaio
    Date:2019-1-15
    Version:1.0
"""

import pandas as pd
import numpy as np


def KNN(x, X, Y, k):
    numSamples = X.shape[0]

    diff = np.tile(x, (numSamples, 1)) - X
    squaredDiff = diff ** 2
    squaredDiffSum = squaredDiff.sum(axis=1)
    distance = squaredDiffSum ** 0.5

    sortedDistIndices = np.argsort(distance)

    classCount = {}
    for i in range(k):
        voteLabel = Y[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    maxCount = 0
    maxIndex = -1
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex  = key

    return maxIndex


def main():
    dataset = pd.read_csv('data.csv')
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, 2].values
    x = [0.9, 1.0]
    k = 2
    classIndex = KNN(x, X, Y, k)
    print('class of x is:', classIndex)

if __name__ == '__main__':
    main()
