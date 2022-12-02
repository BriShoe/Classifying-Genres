import sklearn as sk
import os
import numpy as np
import pandas as pd

def runKNN(dataset, k):
    neigh = sk.neighbors.KNeighborsClassifier(n_neighbors=k)
    X = dataset[dataset.columns.difference(["key", "time_signature", "key_name", "mode_name", "targets"])]
    y = dataset["targets"]
    neigh.fit(X, y)
    return neigh

