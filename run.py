import sklearn as sk
import os
import numpy as np
import pandas as pd

from kNearestNeighbors import runKNN

def randomizeTargets(dataset, numTargets):
    random = np.random.randint(numTargets, size=dataset.shape[0])
    dataset["targets"] = random
    return dataset

def main():
    root = os.getcwd()
    filepath = f"{root}/data/list_of_arctic.csv"
    dataset = randomizeTargets(pd.read_csv(filepath), 3)





