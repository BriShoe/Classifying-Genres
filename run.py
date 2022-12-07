import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import torch
from torch import nn
from train.train_multilabel_nn_pca import make_dataset, fitPCA, prediction_accuracy, multi_classifier
from torch.utils.data import DataLoader

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
    data_p1 = pd.read_csv('data/rock1edited.csv', index_col=0)
    data_p2 = pd.read_csv('data/rock2edited.csv', index_col=0)
    full_train = data_p1.append(data_p2)
    full_test = pd.read_csv('data/rockvalidedited.csv', index_col=0)

    num_genres = 74
    X = full_train.iloc[:, : len(full_train.columns) - num_genres]
    Y = full_train.iloc[:, len(full_train.columns) - num_genres:]

    components = 50
    pca = fitPCA(X, components)
    X = pca.transform(X)

    X_test = full_test.iloc[:, : len(full_test.columns) - num_genres]
    Y_test = full_test.iloc[:, len(full_test.columns) - num_genres:]
    X_test = pca.transform(X_test)

    test_dataset = make_dataset(X_test, Y_test.values)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1)

    model = multi_classifier(X.shape[1], num_genres)
    model.load_state_dict(torch.load(f"models/neuralnetworks/nn_pca_{components}"))

    criterion = nn.BCELoss()

    model.eval()
    test_run_acc = []
    test_run_cost = []
    y_predicts = []
    y_truths = []
    for i, (x_test, y_test) in enumerate(test_dataloader):
        # get predictions
        y_pred = model(x_test)
        y_predicts.extend((torch.max(torch.exp(y_pred), 1)[1]).data.cpu().numpy())
        y_truths.extend(y_test.data.cpu().numpy())

        accuracy = []
        for k, d in enumerate(y_pred, 0):
            acc = prediction_accuracy(torch.Tensor.cpu(y_test[k]), torch.Tensor.cpu(d))
            accuracy.append(acc)
        test_run_acc.append(np.asarray(accuracy).mean())
        cost = criterion(y_pred, y_test)
        test_run_cost.append(cost)
    print('test set')
    print(cost)
    print(np.asarray(test_run_acc).mean())






