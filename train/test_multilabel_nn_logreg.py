#importing libraries
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from archive.test_multilabel_nn import make_dataset, multi_classifier, prediction_accuracy, get_prediction

# evaluate using 10-fold cross-validation
if __name__ == '__main__':
    reducedcolumns = np.loadtxt("../reduce/logregtop100.txt", dtype=str)
    numcolumns = 100

    # combine data
    data_p1 = pd.read_csv('../data/rock1edited_filtered.csv', index_col=0)
    data_p2 = pd.read_csv('../data/rock2edited_filtered.csv', index_col=0)
    full_train = data_p1.append(data_p2)
    full_test = pd.read_csv('../data/rockvalidedited_filtered.csv', index_col=0)
    
    # separate target values
    num_genres = 24
    X = full_train.iloc[:, : len(full_train.columns) - num_genres]
    Y = full_train.iloc[:, len(full_train.columns) - num_genres:]
    X = X[reducedcolumns[:numcolumns]]

    dataset = make_dataset(X, Y.values)
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=32)
    model = multi_classifier(X.shape[1], num_genres)

    # binary cross entropy loss
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # train and output predict loss after every 10 epochs
    epochs = 100
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

    # save model state
    torch.save(model.state_dict(), f"../models/neuralnetworks/nn_logreg_{numcolumns}features")
