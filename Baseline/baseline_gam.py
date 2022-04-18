import numpy as np
import pandas as pd
import sys
import os

dataset = pd.read_pickle('../Dataset/baseline_dataset.gz')
print(dataset.info())

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split

TRAIN_RATIO = 0.9

def separateDataset(dataset, train_ratio):
    '''
    Takes in a dataset (pandas df) and a ratio value, returns a dictionary containing the separated dataset.
    Key "train" = train set, "dev" = dev set (size = train ratio * (sizeof input df - test set)), "test" = test set (size = train ratio * sizeof input df)
    '''
    train_dev_set, test_set = train_test_split(dataset, train_size=train_ratio, random_state=42)
    train_set, dev_set = train_test_split(train_dev_set, train_size=train_ratio, random_state=42)
    print("Training set size: {0}; Dev set size: {1}; Testing set size: {2}".format(len(train_set), len(dev_set), len(test_set)))
    return { "train": train_set, "dev": dev_set, "test": test_set }

def pandasToXY(dataframe):
    '''
    converts the given pandas df to X and Y sub-arrays. X is pandas df, Y is np int array.
    note: the range of columns to select as Y must be double checked when a different dataset is used.
    '''
    X = dataframe.drop(['LOW_IMPACT', 'MID_IMPACT', 'BIG_IMPACT', 'DIRECT_STRIKE'], axis=1)
    Y = np.asarray(dataframe.iloc[:,3:7]).astype(int)
    return X, Y

splitDataset = separateDataset(dataset, TRAIN_RATIO)
train_full = splitDataset["train"]
train_X, train_Y = pandasToXY(train_full)
dev_full = splitDataset["dev"]
dev_X, dev_Y = pandasToXY(dev_full)
test_full = splitDataset["test"]
test_X, test_Y = pandasToXY(test_full)

from pygam import LinearGAM, s

gam0 = LinearGAM(
    s(0)+ 
    s(3, n_splines=400)+s(4, n_splines=400)+s(5, n_splines=400)+
    s(6, n_splines=400)+s(7, n_splines=400)+s(8, n_splines=400)+
    s(9, n_splines=400)+s(10, n_splines=400)+s(11, n_splines=400)+
    s(12, n_splines=400)+s(13, n_splines=400)+s(14, n_splines=400)+
    s(15, n_splines=400)+s(16, n_splines=400)+s(17, n_splines=400),
    max_iter=500, verbose=True
).fit(train_X, train_Y[:,0])
gam0.summary()

gam1 = LinearGAM(
    s(0)+ 
    s(3, n_splines=400)+s(4, n_splines=400)+s(5, n_splines=400)+
    s(6, n_splines=400)+s(7, n_splines=400)+s(8, n_splines=400)+
    s(9, n_splines=400)+s(10, n_splines=400)+s(11, n_splines=400)+
    s(12, n_splines=400)+s(13, n_splines=400)+s(14, n_splines=400)+
    s(15, n_splines=400)+s(16, n_splines=400)+s(17, n_splines=400),
    max_iter=500, verbose=True
).fit(train_X, train_Y[:,1])
gam1.summary()

gam2 = LinearGAM(
    s(0)+ 
    s(3, n_splines=400)+s(4, n_splines=400)+s(5, n_splines=400)+
    s(6, n_splines=400)+s(7, n_splines=400)+s(8, n_splines=400)+
    s(9, n_splines=400)+s(10, n_splines=400)+s(11, n_splines=400)+
    s(12, n_splines=400)+s(13, n_splines=400)+s(14, n_splines=400)+
    s(15, n_splines=400)+s(16, n_splines=400)+s(17, n_splines=400),
    max_iter=500, verbose=True
).fit(train_X, train_Y[:,2])
gam2.summary()


gam3 = LinearGAM(
    s(0)+ 
    s(3, n_splines=400)+s(4, n_splines=400)+s(5, n_splines=400)+
    s(6, n_splines=400)+s(7, n_splines=400)+s(8, n_splines=400)+
    s(9, n_splines=400)+s(10, n_splines=400)+s(11, n_splines=400)+
    s(12, n_splines=400)+s(13, n_splines=400)+s(14, n_splines=400)+
    s(15, n_splines=400)+s(16, n_splines=400)+s(17, n_splines=400),
    max_iter=500, verbose=True
).fit(train_X, train_Y[:,3])
gam3.summary()

raw_preds = np.stack([gam0.predict(dev_X), gam1.predict(dev_X), gam2.predict(dev_X), gam3.predict(dev_X)], axis=1)
best_thresholds = []
best_f1_scores = []
for i in range(4):
    precision, recall, thresholds = precision_recall_curve(dev_Y[:,i], raw_preds[:,i])
    # find best threshold
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.nanargmax(fscore)
    best_thresholds.append(thresholds[ix])
    best_f1_scores.append(fscore[ix])
    print('Best Threshold={0:.05f}, F-Score={1:.05f}'.format(thresholds[ix], fscore[ix]))

preds = []
for i in range(len(raw_preds)):
    pred_row = []
    for j in range(4):
        if raw_preds[i,j] > best_thresholds[j]:
            pred_row.append(1)
        else:
            pred_row.append(0)
    preds.append(pred_row)
preds = np.array(preds)
print(classification_report(dev_Y, preds, digits=5, zero_division=0))

from datetime import datetime
import pickle as pk

model = [gam0, gam1, gam2, gam3]

pk.dump(model, open("./models/baseline_model_gamF_{}.pkl".format(str(datetime.now().strftime("%Y-%m-%d %H-%M"))), 'wb'))

