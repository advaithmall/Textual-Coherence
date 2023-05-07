import numpy as np
import scipy
import pandas as pd
import sys
import csv
import nltk
import re
import pprint
import json
import torch
import argparse
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from model import CoherTagger
#nltk.download('wordnet')
from nltk.corpus import wordnet
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score
from dataset import CoherDataset
stopwords = nltk.corpus.stopwords.words('english')
device = "cuda" if torch.cuda.is_available() else "cpu"

print("using: ", device)


def train(dataset, model, args):
    print("Entered Training...")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    loss_function = nn.NLLLoss()
    accuracy_list = list()
    recall_list = list()
    precision_list = list()
    f1_list = list()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    model.train()
    for epoch in range(args.max_epochs):
        for batch, (sentence, tags) in enumerate(dataloader):
            sentence = torch.tensor(sentence).to(device)
            optimizer.zero_grad()
            y_pred = model(sentence).to(device)
            #print(y_pred.dtype)
            y_pred = y_pred.to(torch.float32)
            #print(y_pred)
            #tags = torch.tensor(tags).to(device)
            #print(tags.dtype, tags.shape, y_pred.shape, y_pred.dtype)
            #print(tags)
            tags = tags.to(torch.int64)
            #print(tags)
            loss = loss_function(y_pred, tags)
            loss.backward()
            optimizer.step()
            pred_list = list()
            for i in range(len(y_pred)):
                pred_list.append(y_pred[i].argmax().item())
            accuary = (y_pred.argmax(1) == tags).float().mean()
            tags = tags.tolist()
            pred_list = pred_list
            recall = recall_score(tags, pred_list, average='macro', zero_division=0)
            precision = precision_score(tags, pred_list, average='macro', zero_division=0)
            f1 = f1_score(tags, pred_list, average='macro', zero_division=0)
            accuracy_list.append(accuary.item())
            recall_list.append(recall)
            precision_list.append(precision)
            f1_list.append(f1)
            print({
                'epoch': epoch, 'batch': batch, 'loss': loss.item(), 'acc': accuary.item(),'f1': f1})
    print("avg acc: ", sum(accuracy_list)/len(accuracy_list), "avg f1: ", sum(f1_list)/len(f1_list),"avg_recall: ", sum(recall_list)/len(recall_list), "avg precision: ", sum(precision_list)/len(precision_list))


def eval(dataset, model, args):
    print("Entered testing...")
    model.eval()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    accuracy_list = list()
    recall_list = list()
    precision_list = list()
    f1_list = list()
    loss_function = nn.NLLLoss()
    for batch, (sentence, tags) in enumerate(dataloader):
        y_pred = model(sentence).to(device)
        y_pred = y_pred.to(torch.float32)
        tags = tags.to(torch.int64)
        loss = loss_function(y_pred, tags)
        pred_list = list()
        for i in range(len(y_pred)):
            pred_list.append(y_pred[i].argmax().item())
        accuary = (y_pred.argmax(1) == tags).float().mean()
        tags = tags.tolist()
        pred_list = pred_list
        recall = recall_score(tags, pred_list, average='macro', zero_division=0)
        precision = precision_score(tags, pred_list, average='macro', zero_division=0)
        f1 = f1_score(tags, pred_list, average='macro', zero_division=0)
        accuracy_list.append(accuary.item())
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)
        print({
            'batch': batch, 'loss': loss.item(), 'acc': accuary.item(),'f1': f1, 'recall': recall, 'precision': precision})
    print("avg acc: ", sum(accuracy_list)/len(accuracy_list), "avg f1: ", sum(f1_list)/len(f1_list),"avg_recall: ", sum(recall_list)/len(recall_list), "avg precision: ", sum(precision_list)/len(precision_list))
        








parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cnn")
parser.add_argument('--max-epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--sequence-length', type=int, default=4)
args = parser.parse_args()
print("Making Dataset...")
temp_set = CoherDataset(args)
torch.save(temp_set, "dataset.pt")
print("Making Model...")
model = CoherTagger(len(temp_set.uniq_words), 2)
model.to(device)
print("Training...")
l1 = int(0.8*len(temp_set))
l2 = len(temp_set) - l1
train_set, test_set = random_split(temp_set, [l1, l2] )
torch.save(train_set, "train_set.pt")
torch.save(test_set, "test_set.pt")
train(train_set, model, args)
torch.save(model, "model.pt")
eval(test_set, model, args)
print("Saving Model...")
