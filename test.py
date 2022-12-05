import numpy as np
from sklearn import datasets
# def calculateLoss():

import seaborn as sns

sns.set_theme()

tips = sns.load_dataset("tips")

sns.relplot(
    data=tips,
    x="total_bill", y="tip", col="time",
    hue="smoker", style="smoker", size="size",
)