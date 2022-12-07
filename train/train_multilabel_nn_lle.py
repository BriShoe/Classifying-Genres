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
from multilabel_cross_val import crossvalidation
from sklearn.manifold import LocallyLinearEmbedding

listOfGenres = sorted([
    'rock---alternative', 'rock---alternativerock', 'rock---bluesrock',
    'rock---britpop', 'rock---classicrock', 'rock---garagerock',
    'rock---glamrock', 'rock---grunge', 'rock---hardrock', 'rock---indie',
    'rock---indiepop', 'rock---indierock', 'rock---newwave', 'rock---poprock',
    'rock---postpunk', 'rock---progressiverock', 'rock---psychedelicrock',
    'rock---punk', 'rock---rockabilly', 'rock---rocknroll',
    'rock---singersongwriter', 'rock---softrock', 'rock---spacerock',
    'rock---stonerrock'
])


# get dataset
class make_dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.length


# get model
class multi_classifier(nn.Module):
    def __init__(self, input_size, neurons, output_size):
        super(multi_classifier, self).__init__()
        self.l1 = nn.Sequential(nn.Linear(input_size, neurons), nn.ReLU(),
                                nn.Dropout(0.5))
        self.l2 = nn.Linear(neurons, output_size)

    def forward(self, x):
        output = self.l1(x)
        output = self.l2(output)
        return F.sigmoid(output)


# train epoch
def train_epoch(model, dataloader, criterion, optimizer):
    train_loss = 0.0
    train_accuracy = []
    model.train()
    for x_train, y_train in dataloader:
        y_pred = model(x_train)
        accuracy = []
        for k, d in enumerate(y_pred, 0):
            acc = prediction_accuracy(torch.Tensor.cpu(y_train[k]),
                                      torch.Tensor.cpu(d))
            accuracy.append(acc)
        train_accuracy.append(np.asarray(accuracy).mean())
        cost = criterion(y_pred, y_train)
        # backprop
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        train_loss += cost.item() * x_train.size(0)  # uncertain about this
    train_acc = np.asarray(train_accuracy).mean()
    return train_loss, train_acc


def valid_epoch(model, dataloader, criterion):
    model.eval()
    valid_loss = 0.0
    valid_accuracy = []
    for x_valid, y_valid in dataloader:
        y_pred = model(x_valid)
        accuracy = []
        for k, d in enumerate(y_pred, 0):
            acc = prediction_accuracy(torch.Tensor.cpu(y_valid[k]),
                                      torch.Tensor.cpu(d))
            accuracy.append(acc)
        valid_accuracy.append(np.asarray(accuracy).mean())
        cost = criterion(y_pred, y_valid)
        valid_loss += cost.item() * x_valid.size(0)  # uncertain about this
    valid_acc = np.asarray(valid_accuracy).mean()
    return valid_loss, valid_acc


# prediction accuracy
def prediction_accuracy(truth, predicted):
    return torch.round(predicted).eq(truth).sum().numpy() / len(truth)


# get single prediction
def get_prediction(x, subgenres, model):
    x_features = torch.tensor(x, dtype=torch.float32)
    res = torch.round(model(x_features))
    res = torch.Tensor.cpu(res).detach().numpy()
    idx = np.argpartition(res, -3)[-3:]

    labels = []
    for i in idx:
        labels.append(subgenres[i])
    print(labels)
    return labels


def print_confusion_matrix(confusion_matrix,
                           axes,
                           class_label,
                           class_names,
                           fontsize=14):
    df_cm = pd.DataFrame(confusion_matrix,
                         index=class_names,
                         columns=class_names)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(),
                                 rotation=0,
                                 ha='right',
                                 fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(),
                                 rotation=45,
                                 ha='right',
                                 fontsize=fontsize)
    axes.set_ylabel('Truth')
    axes.set_xlabel('Predicted')
    axes.set_title(class_label)


def fitLLE(X, k):
    lle = LocallyLinearEmbedding(n_components=k)
    lle.fit(X)
    return lle


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
        lle = fitLLE(X, num)
        X = pd.DataFrame(lle.transform(X))
        crossoutput = crossvalidation(X, Y, [16, 32], [100], [0.001], [32, 64])
        crossoutput["numfeatures"] = num
        print(crossoutput)
        hyperparameters = np.append(hyperparameters, crossoutput)
    hyperparameters = sorted(hyperparameters, key=lambda x: x["f1-score"])
    optimalfeatures = hyperparameters[0]["numfeatures"]
    batchsize, epochs, learningrate, neurons = hyperparameters[0]["hyperparameters"]

    X = full_train.iloc[:, :len(full_train.columns) - num_genres]
    lle = fitLLE(X, optimalfeatures)
    X = pd.DataFrame(lle.transform(X))

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

    with open(f"../models/neuralnetworks/nn_lle", "a") as f:
        f.truncate(0)
        f.write(f"Number of Features: {optimalfeatures} \nBatch Size: {batchsize} \nEpochs: {epochs} \nLearning Rate: {learningrate} \nNeurons: {neurons}")
    torch.save(model.state_dict(), f"../models/neuralnetworks/nn_lle")
