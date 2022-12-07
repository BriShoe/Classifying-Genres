import numpy as np
from multilabel_cross_val import multi_classifier, make_dataset, prediction_accuracy, print_confusion_matrix
import torch
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.manifold import LocallyLinearEmbedding, Isomap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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


def fitIsomap(X, k):
    isomap = Isomap(n_components=k)
    isomap.fit(X)
    return isomap


def fitPCA(X, k):
    pca = PCA(n_components=k)
    pca.fit(X)
    return pca


def fitLLE(X, k):
    lle = LocallyLinearEmbedding(n_components=k)
    lle.fit(X)
    return lle


if __name__ == "__main__":
    modelname = "logreg"

    PATH = f"models/neuralnetworks/nn_{modelname}"
    num_genres = 24
    with open(f"{PATH}.txt") as f:
        lines = f.readlines()
    num_features = int(lines[0][19:])
    neurons = int(lines[4][9:])
    model = multi_classifier(num_features, neurons, num_genres)
    model.load_state_dict(torch.load(f"{PATH}"))

    data_p1 = pd.read_csv('data/rock1edited_filtered.csv', index_col=0)
    data_p2 = pd.read_csv('data/rock2edited_filtered.csv', index_col=0)
    full_train = data_p1.append(data_p2)
    X = full_train.iloc[:, :len(full_train.columns) - num_genres]
    full_test = pd.read_csv("data/rockvalidedited_filtered.csv", index_col=0)
    X_test = full_test.iloc[:, :len(full_test.columns) - num_genres]
    Y_test = full_test.iloc[:, len(full_test.columns) - num_genres:]

    if modelname == "logreg":
        logregcolumns = np.loadtxt("reduce/logregtop100.txt", dtype=str)
        X_test = X_test[logregcolumns[:num_features]]
    elif modelname == "pca":
        pca = fitPCA(X, num_features)
        X_test = pd.DataFrame(pca.transform(X_test))
    elif modelname == "isomap":
        isomap = fitIsomap(X, num_features)
        X_test = pd.DataFrame(isomap.transform(X_test))
    elif modelname == "lle":
        lle = fitLLE(X, num_features)
        X_test = pd.DataFrame(lle.transform(X_test))

    test_dataset = make_dataset(X_test.values, Y_test.values)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 shuffle=False,
                                 batch_size=1)

    criterion = nn.BCELoss()

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

    # Confusion Matrix
    with open(f"results/classification_reports/{modelname}_classification_report.txt", "a") as f:
        f.truncate(0)
        f.write(classification_report(y_truths, y_predicts, target_names=listOfGenres))
    cf_matrix = multilabel_confusion_matrix(y_truths, y_predicts)
    fig, ax = plt.subplots(4, 6, figsize=(12, 7))

    for axes, cfs_matrix, label in zip(ax.flatten(), cf_matrix, listOfGenres):
        print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])

    fig.tight_layout()
    plt.savefig(f'results/cf_matrices/{modelname}_cfmatrix.png')
    plt.show()


