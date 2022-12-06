from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


if __name__ == "__main__":
    selectnum = 10
    numgenres = 74
    data_p1 = pd.read_csv('../data/rock1edited.csv', index_col=0)
    data_p2 = pd.read_csv('../data/rock2edited.csv', index_col=0)
    full_train = data_p1.append(data_p2)

    num_genres = 74
    num_features = len(full_train.columns) - num_genres
    X = full_train.iloc[:, : len(full_train.columns) - num_genres]
    Y = full_train.iloc[:, len(full_train.columns) - num_genres:]

    # fit logregs for each class
    #logregs = [LogisticRegression(random_state=0) for i in range(num_genres)]
    logreg = LogisticRegression(random_state=0)
    genres = Y.columns.values
    #for i in range(len(logregs)):
    genrecolumn = Y[genres[0]]
    logreg.fit(X, genrecolumn)

    weights = np.zeros((num_genres, num_features))

    for i in range(len(logregs)):
        classweights = logregs[i].get_params()
        weights[i] = classweights.values()

    averageweights = np.mean(weights, axis=1)
    weightdict = dict()

    for i in num_features:
        weightdict[X.columns.values()[i]] = averageweights[i]

    rankedfeatures = sorted(weightdict.items(), key = lambda x : x[1], reverse=True)
    topfeatures = np.array(rankedfeatures[:selectnum])[:, 0]

    print(topfeatures)





