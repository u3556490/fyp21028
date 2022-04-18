#!/usr/bin/env python
# coding: utf-8

# # Multilayer Perceptron baseline
# ## Finally, it worked! Breadth > Width

# In[1]:


import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Data

# In[2]:


dataset = pd.read_pickle('../Dataset/baseline_dataset_tsnv_24.gz')
pd.set_option("display.max.columns", None)
print(dataset.info())


# In[8]:


# imports
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, classification_report
from sklearn.model_selection import train_test_split

# Constants
TRAIN_RATIO = 0.9
MAX_ITERS = 1000


# In[4]:


# test-dev-train split

def separateDataset(dataset, train_ratio):
    train_dev_set, test_set = train_test_split(dataset, train_size=train_ratio, random_state=42)
    train_set, dev_set = train_test_split(train_dev_set, train_size=train_ratio, random_state=42)
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


# ### Tools

# In[ ]:


import joblib
from datetime import datetime

def save_sklearn_model_to_file(model, model_type, filename=None):
    if filename == None:
        filename = "./models/baseline_model_{0}_{1}.skl".format(model_type, str(datetime.now().strftime("%Y-%m-%d %H-%M")))
        
    joblib.dump(model, filename)
    
    # to load a model: model = joblib.load(filename)


# In[ ]:


# finds the best decision thresholds and the corresponding F1 scores
# shows the precision-recall curve as well
def optimize_thresholds(clf, datasetX, datasetY):
    all_preds = clf.predict_proba(datasetX)
    best_thresholds = []
    best_f1_scores = []
    n_classes = len(clf.classes_)
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
    n_classes = len(clf.classes_)
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


def regression_label(regr, datasetX, thresholds):
    '''
    Takes in a regressor, a list of decision thresholds, an input samples set X;
    Returns deterministic predictions made using the model over X and the thresholds.
    '''
    preds_probs = np.clip(regr.predict(datasetX),0,1)
    preds = []
    # iterate each predicted probability and compare against threshold
    for i in range(len(preds_probs)):
        pred_row = []
        for j in range(4):
            if preds_probs[i,j] > thresholds[j]:
                pred_row.append(1)
            else:
                pred_row.append(0)
        preds.append(pred_row)
    
    return np.array(preds)


def regressor_find_thresholds(regr, datasetX, datasetY, method='clip'):
    '''
    Takes in a regressor, an input set X, a target set Y and optionally a scaling method;
    returns the best decision thresholds and corresponding f1-scores;
    displays the values and a precision recall curve.
    '''
    all_preds = np.clip(regr.predict(datasetX),0,1)
    best_thresholds = []
    best_f1_scores = []
    for i in range(4):
        precision, recall, thresholds = precision_recall_curve(datasetY[:,i], all_preds[:,i])
        # find best threshold
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.nanargmax(fscore)
        best_thresholds.append(thresholds[ix])
        best_f1_scores.append(fscore[ix])
        print('Best Threshold={0:.05f}, F-Score={1:.05f}'.format(thresholds[ix], fscore[ix]))

    # plot the precision-recall curve for the model
    plt.plot(recall, precision, marker='.', label='PR curve')
    plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Direct Strike')
    plt.legend()
    
    return best_thresholds, best_f1_scores

# In[7]:


# imports
from sklearn.neural_network import MLPClassifier, MLPRegressor


# In[ ]:


mlp_clf = MLPClassifier(
    hidden_layer_sizes=(3072, 1024, 1024, 1024, 256),
    batch_size=512,
    random_state=42,
    max_iter=MAX_ITERS,
    alpha=0.01,
    shuffle=True,
    verbose=True
)

print("[TSNV] MLP Classifier hidden_layer_sizes=(3072, 1024, 1024, 1024, 256), batches 512, regularization alpha=0.01")

mlp_clf = mlp_clf.fit(train_X, train_Y)


# In[ ]:


best_thresholds, best_f1_scores = optimize_thresholds(mlp_clf, dev_X, dev_Y)


# In[ ]:


preds = predictions_with_thresholds(mlp_clf, best_thresholds, dev_X)
print(classification_report(dev_Y, preds, zero_division=0, digits=5))


# In[ ]:


save_sklearn_model_to_file(mlp_clf, "mlpclf")

################## meow!

mlp_regr = MLPRegressor(
    hidden_layer_sizes=(3072, 1024, 1024, 1024, 256),
    batch_size=512,
    random_state=42,
    max_iter=MAX_ITERS,
    alpha=0.01,
    shuffle=True,
    verbose=True
)

print("[TSNV] MLP Regressor hidden_layer_sizes=(3072, 1024, 1024, 1024, 256), batches 512, regularization alpha=0.01")

mlp_regr = mlp_regr.fit(train_X, train_Y)

best_thresholds, best_f1_scores = regressor_find_thresholds(mlp_regr, dev_X, dev_Y)
                                                      
preds = regression_label(mlp_regr, dev_X, best_thresholds)
print(classification_report(dev_Y, preds, zero_division=0, digits=5))   

save_sklearn_model_to_file(mlp_regr, "mlpregr")

