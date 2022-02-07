#!/usr/bin/env python
# coding: utf-8

# In[67]:


"""

QUESTION 1

Some helpful code for getting started.


"""

import numpy as np
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
from imagenet10 import ImageNet10
from imagenet10_testset import ImageNet10_testset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

import pandas as pd
import os
import cv

from config import *

# Gathers the meta data for the images
paths, classes = [], []
for i, dir_ in enumerate(CLASS_LABELS):
    for entry in os.scandir(ROOT_DIR + dir_):
        if (entry.is_file()):
            paths.append(entry.path)
            classes.append(i)
            
data = {
    'path': paths,
    'class': classes
}

data_df = pd.DataFrame(data, columns=['path', 'class'])
data_df = data_df.sample(frac=1).reset_index(drop=True) # Shuffles the data

# Gathers the meta data for the images of test set
paths_test = []
for entry in os.scandir(ROOT_DIR1):
    if (entry.is_file()):
        paths_test.append(entry.path)
            
data_test = {
    'path': paths_test
}

data_df_test = pd.DataFrame(data_test, columns=['path'])
#data_df_test = data_df_test.sample(frac=1).reset_index(drop=True) # Shuffles the data

# See what the dataframe now contains
print("Found", len(data_df), "images.")
print("Found", len(data_df_test), "images.")
# If you want to see the image meta data

print(data_df.head())
print(data_df_test.head())
print(data_df_test.head())



# Split the data into train and test sets and instantiate our new ImageNet10 objects.
train_split = 0.80 # Defines the ratio of train/valid data.

# valid_size = 1.0 - train_size
train_size = int(len(data_df)*train_split)


data_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        #transforms.ColorJitter(hue=0.2, saturation=0.2, brightness=0.2),            # added optimization
        transforms.RandomAffine(degrees=10, translate=(0.1,0.1), scale=(0.9,1.1)),  # added optimization
        transforms.RandomHorizontalFlip(p=0.2),                                     # added optimization        
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD)       
    ])


dataset_train = ImageNet10(
    df=data_df[:train_size],
    transform=data_transform,
)

dataset_valid = ImageNet10(
    df=data_df[train_size:].reset_index(drop=True),
    transform=data_transform,
)

dataset_test = ImageNet10_testset(
    df=data_df_test,
    transform=data_transform,
)

# Data loaders for use during training
train_loader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=16,
    shuffle=True,
    num_workers=2
)

valid_loader = torch.utils.data.DataLoader(
    dataset_valid,
    batch_size=24,
    shuffle=True,
    num_workers=2
)

test_loader = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=24,
    shuffle=False,
    num_workers=2
)

# See what you've loaded
print("len(dataset_train)", len(dataset_train))
print("len(dataset_valid)", len(dataset_valid))
print("len(dataset_test)", len(dataset_test))

print("len(train_loader)", len(train_loader))
print("len(valid_loader)", len(valid_loader))
print("len(test_loader)", len(test_loader))


# labels declartion used in confusion matrix
class_labels = [
  "baboon",
  "banana",
  "canoe",
  "cat",
  "desk",
  "drill",
  "dumbbell",
  "football",
  "mug",
  "orange",
]


# ***Linear model for 1.1.1 question***

# In[ ]:


net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(128*128*3,10)
)

for param in net.parameters():
    print(param.shape)
    


# ***final convolutional model***

# In[69]:


net = nn.Sequential(
    nn.Conv2d(3,32, kernel_size=5, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32,64, kernel_size=3, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout(0.2),
    nn.Conv2d(64,128, kernel_size=3, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout(0.2),
    nn.Conv2d(128,256, kernel_size=3, padding=0),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout(0.3),
    nn.Flatten(),
    nn.Linear(256*6*6,4000),
    nn.ReLU(),
    nn.Linear(4000,900),
    nn.ReLU(),
    nn.Linear(900,90),
    nn.ReLU(),
    nn.Linear(90,10)
)

for param in net.parameters():
    print(param.shape)


# ***save the convolutional model***

# ***path_model="/Users/PG/Documents/AI/CW1/scripts/epoch30_fullset.pth"
# torch.save(net.state_dict(),path_model)
# net.load_state_dict(torch.load(path_model)) 
# net.eval()***

# ***stats function to calculate validation data set loss(used in 1.1.1, 1.1.2, 1.2.1, 1.2.2 questions)***

# In[70]:


def stats(loader, net):
    correct = 0
    total = 0
    running_loss = 0
    n = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            outputs = net(images)
            loss = loss_fn(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels
            running_loss += loss
            n += 1
    return running_loss/n, correct/total 
    #return running_loss/n


# ***Single batch train set epochs(execute two times, one time for linear model and one time for cnn model){used in 1.1.1 and 1.1.2 questions}***

# In[ ]:


nepochs = 30
statsrec_single = np.zeros((3,nepochs))

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

iteration_data=iter(train_loader)                   
inputs, labels = iteration_data.next()            #declare single batch outside for loop

for epoch in range(nepochs):  # loop over the dataset multiple times

    running_loss = 0.0
    n = 0
    optimizer.zero_grad()
    # Forward, backward, and update parameters
    outputs = net(inputs)
    loss = loss_fn(outputs, labels)
    loss.backward()
    optimizer.step()
    
    # accumulate loss
    running_loss += loss.item()
    n += 1
    
    ltrn = running_loss/n
    ltst,atst = stats(valid_loader, net)
    statsrec_single[:,epoch] = (ltrn, ltst,atst)
    print(f"epoch: {epoch} training loss: {ltrn: .3f}  validation loss: {ltst: .3f} Accuracy: {atst: .1%}")


# ***plot for Single batch train set(executed two times, one time for linear model and one time for cnn model){used in 1.1.1, 1.1.2 questions}***

# In[ ]:


fig, ax1 = plt.subplots()
plt.plot(statsrec_single[0], 'r', label = 'training loss', )
plt.plot(statsrec_single[1], 'g', label = 'validation loss' )
plt.legend(loc='center')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and validation loss')
plt.show()


# ***full Train set epochs(execute two times,one time for cnn model(without fine tunning) and one time for cnn model(with fine tuning){ used in 1.2.1, 1.2.2 questions}***

# In[ ]:


nepochs = 30
statsrec_full = np.zeros((3,nepochs))

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


for epoch in range(nepochs):  # loop over the dataset multiple times

    running_loss = 0.0
    n = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        
         # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward, backward, and update parameters
        outputs = net(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
    
        # accumulate loss
        running_loss += loss.item()
        n += 1
    
    ltrn = running_loss/n
    ltst,atst = stats(valid_loader, net)
    statsrec_full[:,epoch] = (ltrn, ltst, atst)
    print(f"epoch: {epoch} training loss: {ltrn: .3f}  validation loss: {ltst: .3f}  Accuracy: {atst: .1%}")


# ***plot for Full data set(executed two times, before and after tunning cnn model){used in 1.2.1, 1.2.2 questions}***

# In[ ]:


fig, ax1 = plt.subplots()
plt.plot(statsrec_full[0], 'r', label = 'training loss', )
plt.plot(statsrec_full[1], 'g', label = 'validation loss' )
plt.legend(loc='center')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and Validation loss, and Accuracy')
ax2=ax1.twinx()
ax2.plot(statsrec_full[2], 'b', label = 'Accuracy')
ax2.set_ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()


# ***confusion matrix function {question 1.2.3}***

# In[ ]:


def stats_confusion_matrix(loader, net):
    correct = 0
    total = 0
    running_loss = 0
    n = 0
    pred_total=torch.empty(0,dtype=int)
    actual_total=torch.empty(0,dtype=int)
    net.eval()
    with torch.no_grad():
        for data in loader:
            images, labels = data
            #images=images.to(device)
            #labels=labels.to(device)
            outputs = net(images)
            #loss = loss_fn(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            pred_total=torch.cat((pred_total,predicted),dim=0)
            actual_total=torch.cat((actual_total,labels),dim=0)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels
            #running_loss += loss
            n += 1
    #return running_loss/n, correct/total 
    #return running_loss/n
    return actual_total,pred_total


# ***execute two below functions for valid and train data set to capture predictions and actual labels{question 1.2.3}***

# In[ ]:


a,b=stats_confusion_matrix(valid_loader,net)
c,d=stats_confusion_matrix(train_loader,net)


# ***confusion matrix plotting{question 1.2.3}***

# In[ ]:


def confusion_mat_train(predicted,actual):
    arr = confusion_matrix(actual.view(-1).detach().cpu().numpy(), predicted.view(-1).detach().cpu().numpy())
    df_mat = pd.DataFrame(arr, class_labels, class_labels)
    plt.figure(figsize = (6,6))
    sns.heatmap(df_mat, annot=True, fmt="d", cmap='YlGnBu')
    plt.xlabel("prediction_labels")
    plt.ylabel("actual_labels(Ground truth)")
    plt.title('Train set confusion matrix')
    plt.show()


# In[ ]:


def confusion_mat_valid(predicted,actual):
    arr = confusion_matrix(actual.view(-1).detach().cpu().numpy(), predicted.view(-1).detach().cpu().numpy())
    df_mat = pd.DataFrame(arr, class_labels, class_labels)
    plt.figure(figsize = (6,6))
    sns.heatmap(df_mat, annot=True, fmt="d", cmap='YlGnBu')
    plt.xlabel("prediction_labels")
    plt.ylabel("actual_labels(Ground truth)")
    plt.title('Validation set confusion matrix')
    plt.show()


# ***execute two below functions for plotting confusion matrix{question 1.2.3}***

# In[ ]:

#invoke the function
confusion_mat_valid(b,a)
confusion_mat_train(d,c)


# ***stats function for test loader to read predictions{question 1.3}***

# In[ ]:


def stats_test_predict(loader, net):
    correct = 0
    total = 0
    running_loss = 0
    n = 0
    pred_total=torch.empty(0,dtype=int)
    net.eval()
    with torch.no_grad():
        for data in loader:
            images = data
            #images=images.to(device)
            #labels=labels.to(device)
            outputs = net(images)
            #loss = loss_fn(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            pred_total=torch.cat((pred_total,predicted),dim=0)
            #total += labels.size(0)    # add in the number of labels in this minibatch
            #correct += (predicted == labels).sum().item()  # add in the number of correct labels
            #running_loss += loss
            n += 1
    #return running_loss/n, correct/total 
    #return running_loss/n
    return pred_total


# ***execute below function to get predictions from test loader{question 1.3}***

# In[ ]:

#invoke the function
e=stats_test_predict(test_loader,net)


# ***function to generate test predictions report csv file{question 1.3}***

# In[155]:


def create_prediction_csv():
    image_path=data_df_test["path"].str.split('/').str[9]
    imagelist=list(image_path)
    preds_list=e.tolist()
    df_pred = pd.DataFrame({'image_name' : imagelist,
                                'id' : preds_list},
                                columns=['image_name','id'])
    with open('[my_student_mm20pgr]_test_preds.csv', 'w',newline="") as csv_file:
        df_pred.to_csv(path_or_buf=csv_file,header=False,index=False)


# ***execute below function to generate csv file for test predictions{question 1.3}***

# In[157]:

#invoke the function
create_prediction_csv()

