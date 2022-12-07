#importing the libraries
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_diabetes
from torch import nn
from reduce.pca_test import test_pca

class net(nn.Module):
    def __init__(self,input_size,output_size):
        super(net,self).__init__()
        self.l1 = nn.Linear(input_size,3)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(3,output_size)
    def forward(self,x):
        output = self.l1(x)
        output = self.relu(output)
        output = self.l2(output)
        return output

#dataset
class make_dataset(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.length = self.x.shape[0]
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
    def __len__(self):
        return self.length

if __name__ == '__main__':
    # load data (x)
    data = pd.read_csv('../data/old/list_of_arctic.csv', index_col=0)
    # random labels (y)
    target = np.array([1 if i % 2 == 0 else 0 for i in range(179)])
    print(data.shape)
    print(target.shape)
    # reduce dim
    pca, reduced_data = test_pca(data, 5)
    print(pca.components_)
    # make dataset and nn
    dataset = make_dataset(reduced_data, target)
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=100)
    model = net(reduced_data.shape[1], 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    # train and output predict loss after every 50 epochs
    epochs = 1500
    costval = []
    for j in range(epochs):
        for i, (x_train, y_train) in enumerate(dataloader):
            # prediction
            y_pred = model(x_train)

            # calculating loss
            cost = criterion(y_pred, y_train.reshape(-1, 1))

            # backprop
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        if j % 50 == 0:
            print(cost)
            costval.append(cost)
