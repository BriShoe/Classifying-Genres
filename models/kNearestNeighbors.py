from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np
import pandas as pd

def KNN(dataset, k):
    neigh = KNeighborsClassifier(n_neighbors=k)
    X = dataset[dataset.columns.difference(["key", "time_signature", "key_name", "mode_name", "target", "track_name", "artist_name", "Unnamed: 0"])]
    y = dataset["genre"]
    neigh.fit(X, y)
    return neigh



