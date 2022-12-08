#importing libraries
import pandas as pd
import torch
import os
import seaborn as sns
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch import nn
import torch.nn.functional as F
from multilabel_cross_val import make_dataset, multi_classifier, crossvalidation, prediction_accuracy

from sklearn.decomposition import PCA


def fitPCA(X, k):
    pca = PCA(n_components=k)
    pca.fit(X)
    return pca


# evaluate using 10-fold cross-validation
if __name__ == '__main__':
    numcolumns = [25, 50, 100]
    # combine data
    data_p1 = pd.read_csv('../data/rock1edited_filtered.csv', index_col=0)
    data_p2 = pd.read_csv('../data/rock2edited_filtered.csv', index_col=0)
    full_train = data_p1.append(data_p2)
    # separate target values and get random splits
    num_genres = 24
    Y = full_train.iloc[:, len(full_train.columns) - num_genres:]

    hyperparameters = np.array([])
    for num in numcolumns:
        X = full_train.iloc[:, :len(full_train.columns) - num_genres]
        pca = fitPCA(X, num)
        X = pd.DataFrame(pca.transform(X))
        crossoutput = crossvalidation(X, Y, [16, 32], [100], [0.001], [(64, 32), (96, 48)])
        crossoutput["numfeatures"] = num
        print(crossoutput)
        hyperparameters = np.append(hyperparameters, crossoutput)
    hyperparameters = sorted(hyperparameters, key=lambda x: x["f1-score"])
    optimalfeatures = hyperparameters[0]["numfeatures"]
    batchsize, epochs, learningrate, neurons = hyperparameters[0]["hyperparameters"]

    X = full_train.iloc[:, :len(full_train.columns) - num_genres]
    pca = fitPCA(X, optimalfeatures)
    X = pd.DataFrame(pca.transform(X))

    dataset = make_dataset(X.values, Y.values)
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=batchsize)
    model = multi_classifier(len(X.columns), neurons, num_genres)
    # binary cross entropy loss
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)

    # train and output predict loss after every 10 epochs
    costval = []
    running_accuracy = []
    for j in range(epochs):
        for i, (x_train, y_train) in enumerate(dataloader):
            # get predictions
            y_pred = model(x_train)
            accuracy = []
            for k, d in enumerate(y_pred, 0):
                acc = prediction_accuracy(torch.Tensor.cpu(y_train[k]), torch.Tensor.cpu(d))
                accuracy.append(acc)
            running_accuracy.append(np.asarray(accuracy).mean())
            cost = criterion(y_pred, y_train)
            # backprop
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        if j % 10 == 0:
            print(cost)
            print(np.asarray(running_accuracy).mean())
            costval.append(cost)

    with open(f"../models/neuralnetworks/nn_pca.txt", "a") as f:
        f.truncate(0)
        f.write(f"Number of Features: {optimalfeatures} \nBatch Size: {batchsize} \nEpochs: {epochs} \nLearning Rate: {learningrate} \nNeurons: {neurons}")
    torch.save(model.state_dict(), f"../models/neuralnetworks/nn_pca")
