#importing libraries
import pandas as pd
import torch
import os
import seaborn as sn
from sklearn.metrics import confusion_matrix
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
def get_prediction(x, subgenres, model):
    x_features = torch.tensor(x,dtype=torch.float32)
    res = torch.round(model(x_features))
    res = torch.Tensor.cpu(res).detach().numpy()
    idx = np.argpartition(res, -3)[-3:]

    labels = []
    for i in idx:
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
    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=32)

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

    # check on one specific case
    listOfGenres = sorted(['rock---alternative', 'rock---indierock', 'rock---singersongwriter',
                    'rock---classicrock', 'rock---poprock', 'rock---progressiverock', 'rock---rockabilly', 'rock---rocknroll',
                    'rock---hardcorepunk', 'rock---punk', 'rock---newwave', 'rock---postpunk', 'rock---alternativerock',
                    'rock---indie', 'rock---hardrock', 'rock---hairmetal', 'rock---artrock', 'rock---bluesrock','rock---alternativepunk',
                    'rock---latinrock', 'rock---powerpop', 'rock---indiepop', 'rock---psychobilly',
                    'rock---stonerrock', 'rock---glamrock', 'rock---aor', 'rock---psychedelicrock', 'rock---britpop', 'rock---newromantic',
                    'rock---emo', 'rock---softrock', 'rock---grunge', 'rock---pianorock', 'rock---american', 'rock---rockabillysoul',
                    'rock---krautrock', 'rock---noisepop', 'rock---stoner', 'rock---garagerock', 'rock---lofi', 'rock---spacerock',
                    'rock---indiefolk', 'rock---alternativemetal', 'rock---guitarvirtuoso', 'rock---powerballad', 'rock---symphonicrock',
                    'rock---rockballad', 'rock---arenarock', 'rock---protopunk', 'rock---numetal', 'rock---rapcore', 'rock---funkrock',
                    'rock---folkpunk', 'rock---surfrock',
                    'rock---anarchopunk', 'rock---stonermetal', 'rock---southernrock', 'rock---poppunk', 'rock---jamband',
                    'rock---funkmetal', 'rock---madchester', 'rock---britishinvasion', 'rock---chamberpop', 'rock---russianrock',
                    'rock---experimentalrock', 'rock---melodicrock', 'rock---postgrunge', 'rock---horrorpunk', 'rock---streetpunk',
                    'rock---jazzrock', 'rock---symphonicprog', 'rock---glam', 'rock---acousticrock',
                    'rock---psychedelicpop'])

    cf_matrix = confusion_matrix(y_truths, y_predicts)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *74, index = [i for i in listOfGenres],
                     columns = [i for i in listOfGenres])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')
    
    X_test = full_test.iloc[1, : len(full_test.columns) - num_genres]
    Y_test = full_test.iloc[1, len(full_test.columns) - num_genres:]
    a = get_prediction(X_test, listOfGenres, model)
    
    idx = np.argpartition(Y_test, -3)[-3:]
    labels = []
    for i in idx:
        labels.append(listOfGenres[i])
    print(labels)


