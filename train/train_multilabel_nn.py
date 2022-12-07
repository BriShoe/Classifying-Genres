#importing libraries
import pandas as pd
import torch
import os
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F

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
    def __init__(self, input_size, output_size):
        super(multi_classifier, self).__init__()
        self.l1 = nn.Sequential(nn.Linear(input_size, 256), nn.ReLU(),
                                nn.Dropout(0.5))
        self.l2 = nn.Linear(256, output_size)

    def forward(self, x):
        output = self.l1(x)
        output = self.l2(output)
        return F.sigmoid(output)


# train epoch
def train_epoch(model, dataloader, criterion, optimizer):
    train_loss = 0.0
    train_accuracy = []
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


# evaluate using 10-fold cross-validation
if __name__ == '__main__':
    # combine data
    data_p1 = pd.read_csv('../data/rock1edited_filtered.csv', index_col=0)
    data_p2 = pd.read_csv('../data/rock2edited_filtered.csv', index_col=0)
    full_train = data_p1.append(data_p2)
    full_test = pd.read_csv('../data/rockvalidedited_filtered.csv',
                            index_col=0)

    # separate target values
    num_genres = 24
    X = full_train.iloc[:, :len(full_train.columns) - num_genres]
    Y = full_train.iloc[:, len(full_train.columns) - num_genres:]

    dataset = make_dataset(X.values, Y.values)
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=32)
    model = multi_classifier(len(full_train.columns) - num_genres, num_genres)
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
                acc = prediction_accuracy(torch.Tensor.cpu(y_train[k]),
                                          torch.Tensor.cpu(d))
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

    # check on test set
    X_test = full_test.iloc[:, :len(full_test.columns) - num_genres]
    Y_test = full_test.iloc[:, len(full_test.columns) - num_genres:]
    test_dataset = make_dataset(X_test.values, Y_test.values)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 shuffle=False,
                                 batch_size=1)

    model.eval()
    test_run_acc = []
    test_run_cost = []
    y_predicts = np.zeros(shape=(len(full_test), num_genres))
    row = 0
    y_truths = Y_test.to_numpy()
    for i, (x_test, y_test) in enumerate(test_dataloader):
        # get predictions
        y_pred = model(x_test)

        res = torch.round(y_pred)
        res = torch.Tensor.cpu(res).detach().numpy()
        y_predicts[row] = res
        row += res.shape[0]

        accuracy = []
        for k, d in enumerate(y_pred, 0):
            acc = prediction_accuracy(torch.Tensor.cpu(y_test[k]),
                                      torch.Tensor.cpu(d))
            accuracy.append(acc)
        test_run_acc.append(np.asarray(accuracy).mean())
        cost = criterion(y_pred, y_test)
        test_run_cost.append(cost)
    print('test set')
    print(cost)
    print(np.asarray(test_run_acc).mean())
    print(y_predicts[0], len(y_predicts))

    #Confusion Matrix
    print(
        classification_report(y_truths, y_predicts, target_names=listOfGenres))
    cf_matrix = multilabel_confusion_matrix(y_truths, y_predicts)
    fig, ax = plt.subplots(4, 6, figsize=(12, 7))

    for axes, cfs_matrix, label in zip(ax.flatten(), cf_matrix, listOfGenres):
        print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])

    fig.tight_layout()
    plt.savefig('visualizations/output.png')
    plt.show()

    # test examples
    X_test = full_test.iloc[69, :len(full_test.columns) - num_genres]
    Y_test = full_test.iloc[69, len(full_test.columns) - num_genres:]
    print('truths', Y_test)
    get_prediction(X_test, listOfGenres, model)

    X_test = full_test.iloc[209, :len(full_test.columns) - num_genres]
    Y_test = full_test.iloc[209, len(full_test.columns) - num_genres:]
    print('truths', Y_test)

    get_prediction(X_test, listOfGenres, model)
