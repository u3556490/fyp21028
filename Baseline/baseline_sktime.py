#!/usr/bin/env python
# coding: utf-8

# # sktime: Direct time series modelling

# In[1]:
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import sys
import os

warnings.simplefilter(action='ignore', category=FutureWarning)

# ### Dataset

# In[2]:


# read Dataset as Pandas DataFrame
dataset = pd.read_pickle('../Dataset/baseline_dataset_ts_24.gz')
pd.set_option("display.max.columns", None)
print(dataset.info())


# In[3]:


# imports
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split

# Constants
TRAIN_RATIO = 0.9


# In[4]:


# test-dev-train split

def separateDataset(dataset, train_ratio):
    train_dev_set, test_set = train_test_split(dataset, train_size=train_ratio)
    train_set, dev_set = train_test_split(train_dev_set, train_size=train_ratio)
    print("Training set size: {0}; Dev set size: {1}; Testing set size: {2}".format(len(train_set), len(dev_set), len(test_set)))
    return { "train": train_set, "dev": dev_set, "test": test_set }

def pandasToXY(dataframe):
    X = dataframe.drop(['LOW_IMPACT', 'MID_IMPACT', 'BIG_IMPACT', 'DIRECT_STRIKE'], axis=1)
    Y = np.asarray(dataframe.iloc[:,0:4]).astype(int)
    return X, Y

# train-dev-test splitting
splitDataset = separateDataset(dataset, TRAIN_RATIO)
# separate each of the 3 sets into X and Y
train_full = splitDataset["train"]
train_X, train_Y = pandasToXY(train_full)
dev_full = splitDataset["dev"]
dev_X, dev_Y = pandasToXY(dev_full)
test_full = splitDataset["test"]
test_X, test_Y = pandasToXY(test_full)


# ### Utilities


#### FOR CLASSIFIERS
# convert predict_proba outputs of (n_targets, n_samples, 2) to (n_samples, n_classes)
def get_multioutput_proba(preds):
    preds = np.array(preds)
    new_preds = []
    for i in range(4):
        new_preds.append(preds[i,:,1])
    new_preds = np.array(new_preds).T
    return new_preds

# finds the best decision thresholds and the corresponding F1 scores
# shows the precision-recall curve as well
def optimize_thresholds(clf, datasetX, datasetY):
    all_preds = clf.predict_proba(datasetX)
    if isinstance(clf, MultiOutputClassifier):
        all_preds = get_multioutput_proba(all_preds)
    best_thresholds = []
    best_f1_scores = []
    n_classes = 4
    for i in range(n_classes):
        precision, recall, thresholds = precision_recall_curve(datasetY[:,i], all_preds[:,i])
        # find best threshold
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.nanargmax(fscore)
        best_thresholds.append(thresholds[ix])
        best_f1_scores.append(fscore[ix])
        print('Best Threshold={0:.05f}, F-Score={1:.05f}'.format(thresholds[ix], fscore[ix]))
        
    return best_thresholds, best_f1_scores

# make predictions according to the given thresholds
def predictions_with_thresholds(clf, thresholds, datasetX):
    preds_probs = clf.predict_proba(datasetX)  
    if isinstance(clf, MultiOutputClassifier):
        preds_probs = get_multioutput_proba(preds_probs)
    n_classes = 4
    preds = []
    # iterate each predicted probability and compare against threshold
    for i in range(len(preds_probs)):
        pred_row = []
        for j in range(n_classes):
            if preds_probs[i,j] > thresholds[j]:
                pred_row.append(1)
            else:
                pred_row.append(0)
        preds.append(pred_row)
    
    return np.array(preds)


# ### Data Prepping

# In[9]:


PAST_TRACK_LIMIT = 24

def convert_X(dataset_X):
    '''Takes in a (n_samples, n_features) Pandas dataframe and returns it in shape (n_samples, n_features, time_series_length)'''
    processed_samples = 0
    new_dataset = []
    for index, row in dataset_X.iterrows():
        new_row = []

        # obtain time series for each feature        
        for i in range(10):
            feature_name = dataset_X.columns[i][:-2]
            feature_series = []
            for j in range(0, PAST_TRACK_LIMIT+6, 6):        
                feature_series.append(row.loc["{0}{1:02d}".format(feature_name, j)]) # access by column name
            feature_series.reverse() # newest data come last
            feature_series = pd.Series(data=feature_series) # correct type for each cell
            new_row.append(feature_series)

        # new_row = pd.Series(data=new_row, index=new_features)
        new_dataset.append(new_row)
        processed_samples += 1

        if processed_samples % 5000 == 0:
            print("Finished concatenating {0}/{1} samples...".format(processed_samples, dataset_X.shape[0]))

    # get new column names
    new_features = []
    for i in range(10):
            feature_name = dataset_X.columns[i][:-2]
            new_features.append(feature_name)
            
    # convert types back
    # converted_X = pd.DataFrame(new_dataset, columns=new_features)
    converted_X = np.array(new_dataset)
    print("Completed")
    return converted_X

train_X = convert_X(train_X)
dev_X = convert_X(dev_X)


# In[10]:


from sktime.transformations.panel.compose import ColumnConcatenator

print("Before transform:", train_X.shape)
concat_train_X = ColumnConcatenator().fit_transform(train_X)
concat_dev_X = ColumnConcatenator().fit_transform(dev_X)
print("After transform:", concat_train_X.shape)
print("Each element is:", concat_train_X.iloc[0].iloc[0].shape)


# In[11]:

from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

from sklearn.multioutput import MultiOutputClassifier


# In[22]:


pt_clf = MultiOutputClassifier(KNeighborsTimeSeriesClassifier(n_neighbors=10,weights='distance',n_jobs=-1), n_jobs=-1).fit(concat_train_X, train_Y)
best_thresholds, best_f1_scores = optimize_thresholds(pt_clf, concat_dev_X, dev_Y)


preds = predictions_with_thresholds(pt_clf, best_thresholds, concat_dev_X)
print(classification_report(dev_Y, preds, zero_division=0, digits=5))
