import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import metrics

from models.kNearestNeighbors import KNN
from models.random_forest import random_forest


def randomizetargets(dataset, numTargets):
    random = np.random.randint(numTargets, size=dataset.shape[0])
    dataset["target"] = random
    return dataset


def standardize(dataset):
    for column in dataset.select_dtypes("float64").columns:
        standcolumn = scale(dataset[column])
        dataset[column] = standcolumn


def convertGenres(dataset):
    genreDict = dict()
    reversegenreDict = dict()
    values = set(dataset["genre"].values)
    for num, genre in enumerate(list(values), start=0):
        genreDict[num] = genre
        reversegenreDict[genre] = num
    dataset["genre"] = dataset["genre"].transform(lambda x: reversegenreDict[x])
    return dataset, genreDict


if __name__ == "__main__":
    reweight = True
    root = os.getcwd()
    filepath = f"{root}/data/songs_concise.csv"
    dataset, genreMap = convertGenres(pd.read_csv(filepath))
    if reweight:
        standardize(dataset)
    train, test = train_test_split(dataset, test_size=0.2, random_state=0)
    model = KNN(train, 1)
    pred = model.predict(test[test.columns.difference(["key", "time_signature", "key_name", "mode_name", "target", "track_name", "artist_name", "Unnamed: 0"])])
    labels = test["genre"].values
    print("Test Error:", np.mean([labels[i] != pred[i] for i in range(len(labels))]))
    model_rf = random_forest(dataset.iloc[:100, :], 100)
    pred_rf = model_rf.predict(dataset[dataset.columns.difference(
        ["key", "time_signature", "key_name", "mode_name", "target", "track_id", "artist_name", "Unnamed: 0"])].iloc[
                         100:, :])
    print("Test Error:", np.mean([labels[i] != pred_rf[i] for i in range(len(labels))]))






