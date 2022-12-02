import sklearn as sk
import os
import numpy as np
import pandas as pd

def runKNN(dataset, k):
    neigh = sk.neighbors.KNeighborsClassifier(n_neighbors=k)
    neigh.fit()

