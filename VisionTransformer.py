
# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

# import the necessary packages
import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import transforms

import cached_dataloader

from sklearn import metrics

from tqdm import tqdm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from transformers import ViTImageProcessor, ViTModel
# this ensures that the current MacOS version is at least 12.3+
print(torch.backends.mps.is_available())
# this ensures that the current current PyTorch installation was built with MPS activated.
print(torch.backends.mps.is_built())

# ************************Metrics*******************************
def accuracy(truth, pred):
    return metrics.accuracy_score(truth, pred)

def precision(y_true, y_pred):
    return metrics.precision_score(y_true, y_pred, average='weighted')

def recall(y_true, y_pred):
    return metrics.recall_score(y_true, y_pred, average='weighted')

def f1score(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred, average='weighted')


# **************************Train*************************************
def train(train_dataset, val_dataset, device, model, criterion, optimizer, lr_scheduler, trial = None):
    EPOCH = 50
    NUM_CLASSES = 4
    REGULARIZATION = False
    REG_LAMBDA = 0.05

    ### Preprocessing Models

    # Image processor
    img_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    best_val_acc = 0

    train_metrics = {"epoch":[],"num_steps":[],"train_loss":[],"val_loss":[],"train_acc":[],"val_acc":[],"train_precision":[],"val_precision":[],"train_recall":[],"val_recall":[],"train_f1score":[],"val_f1score":[]}
    for epoch in range(1, EPOCH+1):

        #  Storage variables
        num_steps = 0
        val_num_steps = 0
        running_loss = 0.0
        val_running_loss = 0.0
        true = []
        pred = []
        val_true = []
        val_pred = []

        for i, (data, target) in tqdm(enumerate(train_dataset),total=len(train_dataset), desc=f"[Epoch {epoch}]",ascii=' >='):            
            img_batch = []
            labels = target.to(device)
            labels_one_hot = F.one_hot(labels.to(torch.int64).squeeze(), NUM_CLASSES)
            img_batch = [img for img in data]
            img_processed = img_processor(img_batch, return_tensors='pt').to(device) 
            outputs = model(img_processed).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            if(REGULARIZATION == True):
                # add L2 regularization to the loss function
                regularization_loss = 0
                for param in model.parameters():
                    regularization_loss += torch.sum(torch.square(param))
                
                loss = criterion(outputs, labels_one_hot.to(torch.float32)) + REG_LAMBDA * regularization_loss
            else:
                loss = criterion(outputs, labels_one_hot.to(torch.float32))

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            true.extend(labels_one_hot.cpu().detach().numpy())
            pred.extend(outputs.cpu().detach().numpy())
           
            num_steps +=1

        
        with torch.no_grad():
            for j, (val_data, val_target) in tqdm(enumerate(val_dataset), total = len(val_dataset), desc=f"[Epoch {epoch}]",ascii=' >='):

                val_img_batch = []

                val_labels = val_target.to(device)
                val_labels_one_hot = F.one_hot(val_labels.to(torch.int64).squeeze(), NUM_CLASSES)
                
                val_img_batch = [val_img for val_img in val_data]
                img_processed = img_processor(val_img_batch, return_tensors='pt').to(device) 

                val_outputs = model(img_processed).to(device)

                val_loss = criterion(val_outputs, val_labels_one_hot.to(torch.float32))
                val_true.extend(val_labels_one_hot.cpu().detach().numpy())
                val_running_loss += val_loss.item()

                val_pred.extend(val_outputs.cpu().detach().numpy())
                
                val_num_steps +=1
        
        lr_scheduler.step()
        
        print("Unique ******************* ", np.unique(np.argmax(true, axis=1)))
        train_acc = accuracy(np.argmax(true, axis=1), np.argmax(pred, axis=1))
        val_acc = accuracy(np.argmax(val_true, axis=1), np.argmax(val_pred, axis=1))

        train_precision = precision(np.argmax(true, axis=1), np.argmax(pred, axis=1))
        val_precision = precision(np.argmax(val_true, axis=1), np.argmax(val_pred, axis=1))

        train_recall = recall(np.argmax(true, axis=1), np.argmax(pred, axis=1))
        val_recall = recall(np.argmax(val_true, axis=1), np.argmax(val_pred, axis=1))

        train_f1 = f1score(np.argmax(true, axis=1), np.argmax(pred, axis=1))
        val_f1 = f1score(np.argmax(val_true, axis=1), np.argmax(val_pred, axis=1))

        print(f'Num_steps : {num_steps}, train_loss : {running_loss/num_steps:.3f}, val_loss : {val_running_loss/val_num_steps:.3f}, train_acc : {train_acc}, val_acc : {val_acc}, train_f1score : {train_f1}, val_f1score : {val_f1}')

        if(val_acc > best_val_acc):
            best_val_acc = val_acc
            best_model = model

        train_metrics["epoch"].append(epoch)
        train_metrics["num_steps"].append(num_steps)
        train_metrics["train_loss"].append(running_loss/num_steps)
        train_metrics["val_loss"].append(val_running_loss/val_num_steps)
        train_metrics["train_acc"].append(train_acc)
        train_metrics["val_acc"].append(val_acc)
        train_metrics["train_precision"].append(train_precision)
        train_metrics["val_precision"].append(val_precision)
        train_metrics["train_recall"].append(train_recall)
        train_metrics["val_recall"].append(val_recall)
        train_metrics["train_f1score"].append(train_f1)
        train_metrics["val_f1score"].append(val_f1)

    return best_model, train_metrics

# ************************* MODEL DEFINITION **********************************
class MLP(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_sizes=[128, 64], dropout_probability=[0.5,0.7]):
        super(MLP, self).__init__()
        assert len(hidden_sizes) >= 1 , "specify at least one hidden layer"
        
        self.layers = self.create_layers(in_channels, num_classes, hidden_sizes, dropout_probability)


    def create_layers(self, in_channels, num_classes, hidden_sizes, dropout_probability):
        layers = []
        layer_sizes = [in_channels] + hidden_sizes + [num_classes]
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_probability[i]))
            else:
                layers.append(nn.Softmax(dim=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.layers(out)
        return out

class VisionModel(nn.Module):
    def __init__(self, modality1, mlp_hidden_sizes, dropout_prob, batch_size, device):
        super().__init__()
        self.device = device
        self.batch_size = batch_size

        self.modality1 = modality1.to(self.device)
        self.mlp_hidden_sizes = mlp_hidden_sizes
        self.dropout_prob = dropout_prob
        self.head = MLP(in_channels=self._calculate_in_features(),
                            num_classes=4,
                            hidden_sizes=self.mlp_hidden_sizes, 
                            dropout_probability= self.dropout_prob).to(self.device)


        for param in self.modality1.parameters():
            param.requires_grad = True

        for param in self.head.parameters():
            param.requires_grad = True

    def forward(self, input1):
        image_output = self.modality1(**input1)['last_hidden_state'].to(self.device)
        image_output = torch.nn.Flatten()(image_output).to(self.device)
        head_output = self.head(image_output).to(self.device)
        return head_output
    
    def _calculate_in_features(self):
        # Create an example input and pass it through the network to get the output size
        img_batch = []
        img = torch.randint(0, 255, size=(self.batch_size, 3, 224, 224)).float()
        img_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        img_batch = [each for each in img]
        input1 = img_processor(img_batch, return_tensors='pt').to(device) 
        image_output = self.modality1(**input1)['last_hidden_state'].to(self.device)
        image_output = torch.nn.Flatten()(image_output).to(self.device)
        return image_output.shape[1]

# **************************MAIN***************************************
# torch.manual_seed(25)

# global variables
BATCH_SIZE = 64
TRAIN_SPLIT = 0.9
MLP_HIDDEN_SIZES = [1024,512,256]
DROPOUT_PROB = [0, 0, 0]
LR = 0.1
MOMENTUM = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print(device)
train_dataset, val_dataset = cached_dataloader.getData(BATCH_SIZE, TRAIN_SPLIT)

for x in train_dataset:
    print(type(x))
    break

for x in val_dataset:
    print(type(x))
    break

modality1 = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
model = VisionModel(modality1, MLP_HIDDEN_SIZES, DROPOUT_PROB, BATCH_SIZE, device)

# define the optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=0.0005)
optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
# define the loss
criterion = torch.nn.CrossEntropyLoss()
# define the learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

best_model, train_metrics = train(train_dataset, val_dataset, device, model, criterion, optimizer, lr_scheduler)
metrics_df = pd.DataFrame(train_metrics)
metrics_df.to_csv("Metrics.csv")