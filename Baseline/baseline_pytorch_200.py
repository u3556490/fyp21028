#!/usr/bin/env python
# coding: utf-8

# # Baseline Model - PyTorch version
# 
# Doesn't work, if at all. Dependencies:
# ```
# - numpy
# - pandas
# - torch (with CUDA support and torchvision)
# - torchmetrics
# - matplotlib
# - seaborn
# - sklearn
# ```
# 
# ### Importing libraries and reading files

# In[1]:


import os
os.environ['CUDA_LAUNCH_BLOCKING']="1" # CUDA debug help
# Import statements
import numpy as np
import pandas as pd
import torch
import sys
import matplotlib as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Check GPU and CPU count
print("================ GPU availability, CPU count and GPU status check ===============")
print(torch.cuda.get_device_name() if torch.cuda.is_available() else "No GPU")
print(os.cpu_count())
get_ipython().system('nvidia-smi')


# In[3]:


# Set path (optional)
'''dataset_dir = '../Dataset'
sys.path.append(dataset_dir)'''
get_ipython().system('pwd')


# In[4]:


# read Dataset as Pandas DataFrame
print("================ Reading dataset, null check ===============")
dataset = pd.read_pickle('../Dataset/baseline_dataset.gz')
pd.set_option("display.max.columns", None)
print(dataset.info())


# In[5]:


# san-check: any nulls?
print("Null checking:")
print(dataset[dataset.isnull().any(axis=1)])


# In[6]:


# dataset.describe()


# ### Prepare dataset for model to read

# In[6]:


# constants
TRAIN_RATIO = 0.85
NUM_EPOCHS = 200
LR = 0.001
BATCH_SIZE = 256


# In[7]:


# test-dev-train split
from sklearn.model_selection import train_test_split

def separateDataset(dataset, train_ratio):
    train_dev_set, test_set = train_test_split(dataset, train_size=train_ratio)
    train_set, dev_set = train_test_split(train_dev_set, train_size=0.8)
    print("Training set size: {0}; Dev set size: {1}; Testing set size: {2}".format(len(train_set), len(dev_set), len(test_set)))
    return { "train": train_set, "dev": dev_set, "test": test_set }


# In[8]:


from torch.utils.data import Dataset, DataLoader

# Dataset class to feed into the model
class BaselineDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.drop(['LOW_IMPACT', 'MID_IMPACT', 'BIG_IMPACT', 'DIRECT_STRIKE'], axis=1)
        self.labels = np.asarray(dataframe.iloc[:,3:7]).astype(float)
  
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        rowToUse = torch.tensor(self.df.iloc[idx]).float()
        labelVector = torch.tensor(self.labels[idx]).float()
        return (rowToUse, labelVector)


# In[9]:


splitDataset = separateDataset(dataset, TRAIN_RATIO)

train_set = BaselineDataset(splitDataset["train"])
dev_set = BaselineDataset(splitDataset["dev"])
test_set = BaselineDataset(splitDataset["test"])

train_dl = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
dev_dl = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_dl = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# ### Model definition

# In[11]:


import torch.nn as nn

class BaselineMLP(nn.Module):
    def __init__(self):
        super(BaselineMLP, self).__init__()
        
        # input shape: (batch_size, 18)
        # output shape: (batch_size, 4)
        
        self.fc = nn.Sequential(
            nn.Linear(18, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4), 
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x


# ### Training Procedures

# In[12]:


from sklearn.metrics import accuracy_score, f1_score
import torchmetrics

# calculate accuracies and such
def accuracy(output, target):
    with torch.no_grad():
        #batch_size = output.size(0)
        
        target = target.cpu()
        preds = torch.round(output).cpu()
        acc = accuracy_score(target, preds)
        # acc = torchmetrics.functional.accuracy(preds, target.int(), num_classes=2, threshold=0.5, multiclass=True)        
        f1 = f1_score(target, preds, average=None, zero_division=0)
    
    return acc, f1


# In[13]:


# train for one epoch

def train_model_for_one_epoch(model, train_dl, criterion, optimizer):
    
    model.train()
    
    running_loss = 0.0
    running_acc = 0.0
    running_f1 = np.array([0.0, 0.0, 0.0, 0.0])
        
    for i, (inputs, targets) in enumerate(train_dl):
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        # training steps
        optimizer.zero_grad()
        y = model(inputs)
        loss = criterion(y, targets)
        loss.backward()
        optimizer.step()

        # evaluate accuracy during training
        acc, f1 = accuracy(y, targets)
        
        # book-keeping: average this epoch
        running_loss += loss.item()
        running_acc += acc
        running_f1 += f1
    
    average_loss = running_loss/len(train_set)
    average_acc = running_acc/len(train_set)
    average_f1 = np.divide(running_f1, len(train_set))
    
    return average_acc, average_f1, average_loss


# In[14]:


# evaluate on dev set

def evaluate(model, dev_dl, criterion):
    
    model.eval()
    
    running_loss = 0.0
    running_acc = 0.0
    running_f1 = np.array([0.0, 0.0, 0.0, 0.0])
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dev_dl):
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            y = model(inputs)
            loss = criterion(y, targets)
            
            acc, f1 = accuracy(y, targets)
            
            running_loss += loss.item()
            running_acc += acc
            running_f1 += f1
            
    return running_acc/len(dev_set), np.divide(running_f1, len(dev_set)), running_loss/len(dev_set)


# In[15]:


# main training loop
from torch.optim import Adam, SGD
import time

def main_training_loop(model, train_dl, dev_dl):
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=LR)
    #optimizer = SGD(model.parameters(), lr=LR)
    
    train_acc_history = []
    train_loss_history = []
    dev_acc_history = []
    dev_loss_history = []
    
    model.cuda()
        
    start_time = time.time()
    print("Main training loop is starting.")
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        
        train_acc, train_f1, train_loss = train_model_for_one_epoch(model, train_dl, criterion, optimizer)
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)
        
        dev_acc, dev_f1, dev_loss = evaluate(model, dev_dl, criterion)
        dev_acc_history.append(dev_acc)
        dev_loss_history.append(dev_loss)
        
        epoch_time = time.time() - epoch_start
        
        if epoch % 1 == 0:
            print("Progress: {0}/{1} epochs, train acc = {2:.5f}, dev acc = {3:.5f}, train loss = {4:.5f}, epoch time = {5:.2f}s".format(epoch+1, NUM_EPOCHS, train_acc, dev_acc, train_loss, epoch_time))

    training_time = time.time() - start_time
    print("Training has completed. Time elapsed = {0}s".format(training_time))
    
    training_history = { 'loss': train_loss_history, 'acc': train_acc_history }
    validate_history = { 'loss': dev_loss_history, 'acc': dev_acc_history }
    
    return training_history, validate_history


# In[16]:


# testing on test set
from sklearn.metrics import precision_score, recall_score

def test_model(model, test_dl):
    criterion = nn.BCELoss()
    
    model.eval()
    
    running_loss = 0.0
    running_acc = 0.0
    running_f1 = np.array([0.0, 0.0, 0.0, 0.0])
    running_precision = np.array([0.0, 0.0, 0.0, 0.0])
    running_recall = np.array([0.0, 0.0, 0.0, 0.0])
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_dl):
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            y = model(inputs)
            loss = criterion(y, targets)
            
            acc, f1 = accuracy(y, targets)
            
            running_loss += loss.item()
            running_acc += acc
            running_f1 += f1
            
            y = torch.round(y).cpu()
            targets = targets.cpu()
            running_precision += precision_score(targets, y, average=None, zero_division=0)
            running_recall += recall_score(targets, y, average=None, zero_division=0)            
            
    final_avg_acc = running_acc/len(test_set)
    final_avg_loss = running_loss/len(test_set)
    final_avg_f1 = np.divide(running_f1, len(test_set))
    final_avg_precision = np.divide(running_precision, len(test_set))
    final_avg_recall = np.divide(running_recall, len(test_set))
    
    return {'acc': final_avg_acc, 'f1': final_avg_f1, 'loss': final_avg_loss, 'precision': final_avg_precision, 'recall': final_avg_recall}


# In[17]:


print("================ Main training and evaluation procedure ===============")
model = BaselineMLP()
training_history, validate_history = main_training_loop(model, train_dl, dev_dl)
results = test_model(model, test_dl)
print(results)


# #### Saving training results to file

# In[15]:


# save model as checkpoint
print("================ Saving results to file ===============")

from datetime import datetime
torch.save({
    'model_state_dict': model.state_dict(),
}, "./models/baseline_model_{0}.pt".format(str(datetime.now().strftime("%Y-%m-%d %H:%M"))))

# save histories to file
df = pd.DataFrame.from_dict(training_history)
df.to_csv("./models/baseline_model_{0}_training.csv".format(str(datetime.now().strftime("%Y-%m-%d %H:%M"))))
df = pd.DataFrame.from_dict(validate_history)
df.to_csv("./models/baseline_model_{0}_validate.csv".format(str(datetime.now().strftime("%Y-%m-%d %H:%M"))))

# save scores to file
array_like_scores = { key: value for key, value in results.items() if key in ["f1", "precision", "recall"] }
df = pd.DataFrame.from_dict(array_like_scores)
df.to_csv("./models/baseline_model_{0}_array_scores.csv".format(str(datetime.now().strftime("%Y-%m-%d %H:%M"))))
general_scores = { key: [value] for key, value in results.items() if key in ["acc", "loss"] }
df = pd.DataFrame.from_dict(general_scores)
df.to_csv("./models/baseline_model_{0}_general_scores.csv".format(str(datetime.now().strftime("%Y-%m-%d %H:%M"))))

