import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from models.kNearestNeighbors import KNN


def randomizetargets(dataset, numTargets):
    random = np.random.randint(numTargets, size=dataset.shape[0])
    dataset["target"] = random
    return dataset


def standardize(dataset):
    for column in dataset.select_dtypes("float64").columns:
        standcolumn = scale(dataset[column])
        dataset[column] = standcolumn


if __name__ == "__main__":
    reweight = True
    root = os.getcwd()
    filepath = f"{root}/data/list_of_arctic.csv"
    dataset = randomizetargets(pd.read_csv(filepath), 3)
    if reweight:
        standardize(dataset)
    model = KNN(dataset.iloc[:100, :], 5)
    pred = model.predict(dataset[dataset.columns.difference(["key", "time_signature", "key_name", "mode_name", "target", "track_id", "artist_name", "Unnamed: 0"])].iloc[100:, :])
    labels = dataset["target"][100:].values
    print("Test Error:", np.mean([labels[i] != pred[i] for i in range(len(labels))]))






