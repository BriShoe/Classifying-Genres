#importing libraries
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F


# get dataset


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

# predict for single track (unfinished)
def predict(x, subgenres, model):
    op = model(x)
    op_b = torch.round(op)
    op_b_np = torch.Tensor.cpu(op_b).detach().numpy()
    preds = np.where(op_b_np == 1)[1]
    sigs_op = torch.Tensor.cpu(torch.round((op)*100)).detach().numpy()[0]
    o_p = np.argsort(torch.Tensor.cpu(op).detach().numpy())[0][::-1]
    label = []
    for i in preds:
        label.append(subgenres[i])
    arg_s = {}
    for i in o_p:
        arg_s[subgenres[int(i)]] = sigs_op[int(i)]
    return label, list(arg_s.items())[:10]
    
# evaluate using 10-fold cross-validation
if __name__ == '__main__':
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=32)
    model = multi_classifier()
    # binary cross entropy loss
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # train and output predict loss after every 10 epochs
    epochs = 100
    costval = []
    for j in range(epochs):
        for i, (x_train, y_train) in enumerate(dataloader):
            # calculate loss
            y_pred = model(x_train)
            cost = criterion(y_pred, y_train.reshape(-1, 1))
            # backprop
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        if j % 50 == 0:
            print(cost)
            costval.append(cost)
