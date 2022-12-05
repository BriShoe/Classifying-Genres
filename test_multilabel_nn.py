#importing libraries
import pandas as pd
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F


# get dataset
class make_dataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    def __len__(self):
        return self.length

# get model
class multi_classifier(nn.Module):
    def __init__(self,input_size,output_size):
        super(multi_classifier,self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_size,256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.l2 = nn.Linear(256,output_size)
    def forward(self,x):
        output = self.l1(x)
        output = self.l2(output)
        return F.sigmoid(output)

# prediction accuracy
def prediction_accuracy(truth, predicted):
    return torch.round(predicted).eq(truth).sum().numpy()/len(truth)


# get single prediction
def get_prediction(x, subgenres):
    predict = torch.round(model(x)).numpy()
    labels = []
    for i in range(len(predict)):
        if predict[i] == 1:
            labels.append(subgenres[i])
    print(labels)
    return labels
    


# evaluate using 10-fold cross-validation
if __name__ == '__main__':
    # combine data
    data_p1 = pd.read_csv('data/rock1edited.csv', index_col = 0)
    data_p2 = pd.read_csv('data/rock2edited.csv', index_col = 0)
    full_train = data_p1.append(data_p2)
    full_test = pd.read_csv('data/rockvalidedited.csv', index_col = 0)
    
    # separate target values
    num_genres = 74
    X = full_train.iloc[:, : len(full_train.columns) - num_genres]
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
    
    # check on test set
    X_test = full_test.iloc[:, : len(full_test.columns) - num_genres]
    Y_test = full_test.iloc[:, len(full_test.columns) - num_genres:]
    test_dataset = make_dataset(X_test.values, Y_test.values)
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=True, batch_size=32)

    model.eval()
    test_run_acc = []
    test_run_cost = []
    for i, (x_test, y_test) in enumerate(test_dataloader):
        # get predictions
        y_pred = model(x_test)
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


