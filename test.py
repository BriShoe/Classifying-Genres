import numpy as np
from sklearn import datasets
# def calculateLoss():

import seaborn as sns


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

if __name__ == "__main__":
    model = multi_classifer
    model.load_state_dict(torch.load(PATH))
    model.eval()

    Y_test = readcsv
    X_test = readcsv

    y_predicts, y_truths