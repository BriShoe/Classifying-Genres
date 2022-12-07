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


# print confusion matrix
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
def crossvalidation(X, Y, batchsizes, epochs, learningrates, neurons):
    # combine data
    performance = dict()
    for variableset in [(batchsize, epoch, learningrate, neuron) for batchsize in batchsizes
                        for epoch in epochs for learningrate in learningrates for neuron in neurons]:
        # separate target values and get random splits
        num_genres = 24

        dataset = make_dataset(X.values, Y.values)
        splits = KFold(n_splits=10, shuffle=True, random_state=0)
        batch_size = variableset[0]

        # train over folds
        history = {
            'train_loss': [],
            'test_loss': [],
            'train_acc': [],
            'test_acc': []
        }
        y_predicts = np.empty(shape=(len(X), num_genres))
        y_truths = np.empty(shape=(len(X), num_genres))
        row = 0
        for fold, (train_idx,
                   val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
            print('Fold {}'.format(fold + 1))
            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      sampler=train_sampler)
            test_loader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     sampler=test_sampler)

            model = multi_classifier(
                len(X.columns), variableset[3], num_genres)
            # binary cross entropy loss
            criterion = nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=variableset[2])

            for epoch in range(variableset[1]):
                train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                                    optimizer)
                test_loss, test_acc = valid_epoch(model, test_loader, criterion)

                train_loss = train_loss / len(train_loader.sampler)
                test_loss = test_loss / len(test_loader.sampler)

                if (epoch + 1) % 10 == 0:
                    print(
                        "Epoch: {}/{} AVG Training Loss: {:.3f} AVG Test Loss: {:.3f} AVG Training Acc {:.6f} AVG Test Acc {:.6f} %"
                        .format(epoch + 1, 100, train_loss, test_loss, train_acc,
                                test_acc))
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

            for x_test, y_test in test_loader:
                y_pred = model(x_test)
                res = torch.round(y_pred)
                res = torch.Tensor.cpu(res).detach().numpy()
                truths = torch.Tensor.cpu(y_test).detach().numpy()
                for i in range(res.shape[0]):
                    y_predicts[row] = res[i]
                    y_truths[row] = truths[i]
                    row += 1

        # train and output predict loss after every 10 epochs
        avg_train_loss = np.mean(history['train_loss'])
        avg_test_loss = np.mean(history['test_loss'])
        avg_train_acc = np.mean(history['train_acc'])
        avg_test_acc = np.mean(history['test_acc'])
        print('Performance of {} fold cross validation'.format(10))
        print(
            "Average Training Loss: {:.4f} \t Average Test Loss: {:.4f} \t Average Training Acc: {:.6f} \t Average Test Acc: {:.6f}"
            .format(avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc))
        classreport = classification_report(y_truths, y_predicts, target_names=listOfGenres, output_dict=True)
        performance[variableset] = classreport["micro avg"]["f1-score"]
    performances = sorted(performance.items(), key=lambda x: x[1], reverse=True)
    print(performances)
    return performances[0][0]

if __name__ == "__main__":
    data_p1 = pd.read_csv('data/rock1edited_filtered.csv', index_col=0)
    data_p2 = pd.read_csv('data/rock2edited_filtered.csv', index_col=0)
    full_train = data_p1.append(data_p2)

    num_genres = 24

    X = full_train.iloc[:, :len(full_train.columns) - num_genres]
    Y = full_train.iloc[:, len(full_train.columns) - num_genres:]

    print("Best Hyperparamters: ", crossvalidation(X, Y, [16, 32], [10], [0.001], [256]))

    # save model
    #torch.save(model.state_dict(), f"models/neuralnetworks/nn_baseline")

    # get confusion matrix and precision/recall metrics
    """print('bluuuuu')
    print(y_predicts[0], len(y_predicts), y_truths[0])
    
    #Confusion Matrix
    report = classification_report(y_truths,
                                       y_predicts,
                                       target_names=listOfGenres)
    print(report)
    with open("results/classification_reports/nn_classification_report.txt",
                  "a") as f:
        f.write(report)
    
    cf_matrix = multilabel_confusion_matrix(y_truths, y_predicts)
    fig, ax = plt.subplots(4, 6, figsize=(12, 7))
    
    for axes, cfs_matrix, label in zip(ax.flatten(), cf_matrix, listOfGenres):
        print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])
    
    fig.tight_layout()
    plt.savefig('visualizations/nn_cross_val_output.png')
    plt.show()
    
    X_test = full_test.iloc[1, :len(full_test.columns) - num_genres]
    Y_test = full_test.iloc[1, len(full_test.columns) - num_genres:]
    get_prediction(X_test, listOfGenres, model)
    
    idx = np.argpartition(Y_test, -3)[-3:]
    labels = []
    for i in idx:
        labels.append(listOfGenres[i])
    print(labels)"""
