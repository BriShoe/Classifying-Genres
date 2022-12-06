from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from test_multilabel_nn import print_confusion_matrix, classification_report


if __name__ == "__main__":
    selectnum = 100
    data_p1 = pd.read_csv('../data/rock1edited_filtered.csv', index_col=0)
    data_p2 = pd.read_csv('../data/rock2edited_filtered.csv', index_col=0)
    full_train = data_p1.append(data_p2)
    full_test = pd.read_csv('../data/rockvalidedited_filtered.csv', index_col=0)

    num_genres = 24
    num_features = len(full_train.columns) - num_genres
    X = full_train.iloc[:, : len(full_train.columns) - num_genres]
    Y = full_train.iloc[:, len(full_train.columns) - num_genres:]
    X_test = full_test.iloc[:, : len(full_test.columns) - num_genres]
    Y_test = full_test.iloc[:, len(full_test.columns) - num_genres:]

    fig, ax = plt.subplots(4, 6, figsize=(12, 7))

    # fit logregs for each class
    totalYpred = np.array([])
    logregs = [LogisticRegression(random_state=0, C=1e5) for i in range(num_genres)]
    genres = Y.columns.values
    for i, axes in zip(range(len(logregs)), ax.flatten()):
        genrecolumn = Y[genres[i]]
        testgenrecolumn = Y_test[genres[i]]
        logregs[i].fit(X, genrecolumn)
        Y_pred = logregs[i].predict(X_test)
        totalYpred = np.append(totalYpred, Y_pred)
        print(f"Logreg train accuracy for {genres[i]}: ", logregs[i].score(X, genrecolumn))
        print(f"Logreg test accuracy for {genres[i]}: ", logregs[i].score(X_test, testgenrecolumn))
        cf_matrix = confusion_matrix(testgenrecolumn, Y_pred)
        print_confusion_matrix(cf_matrix, axes, genres[i], ["N", "Y"])

    totalYpred = np.reshape(totalYpred, (1862, 24)).astype(int)
    print(Y_test.shape, totalYpred.shape)

    with open("logregclassification_report.txt", "a") as f:
        f.write(classification_report(Y_test, totalYpred, target_names=genres))

    fig.tight_layout()
    plt.savefig('../visualizations/logreg_cfmatrix.png')
    plt.show()


    weights = np.zeros((num_genres, num_features))

    for i in range(len(logregs)):
        weights[i] = logregs[i].coef_
        print(f"Weights for class {i}: ", weights[i])

    print(weights.shape)

    averageweights = np.mean(abs(weights), axis=0)
    weightdict = dict()

    print(X.columns.values)
    print(averageweights)
    for i in range(num_features):
        weightdict[X.columns.values[i]] = averageweights[i]

    print(weightdict)

    rankedfeatures = sorted(weightdict.items(), key=lambda x: x[1], reverse=True)
    topfeatures = np.array(rankedfeatures[:selectnum])[:, 0]

    with open('logregtop100.txt', "a") as f:
        for feature in topfeatures:
            f.write(feature)
            f.write("\n")





