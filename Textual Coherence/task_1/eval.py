import numpy as np
import scipy
import pandas as pd
import sys
import csv
import nltk
#import re
#import pprint
#import json
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
#stopwords = nltk.corpus.stopwords.words('english')
device = "cuda" if torch.cuda.is_available() else "cpu"

print("using: ", device)


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
        recall = recall_score(
            tags, pred_list, average='macro', zero_division=0)
        precision = precision_score(
            tags, pred_list, average='macro', zero_division=0)
        f1 = f1_score(tags, pred_list, average='macro', zero_division=0)
        accuracy_list.append(accuary.item())
        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)
        print({
            'batch': batch, 'loss': loss.item(), 'acc': accuary.item(), 'f1': f1, 'recall': recall, 'precision': precision})
    print("avg acc: ", sum(accuracy_list)/len(accuracy_list), "avg f1: ", sum(f1_list)/len(f1_list),
          "avg_recall: ", sum(recall_list)/len(recall_list), "avg precision: ", sum(precision_list)/len(precision_list))


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="cnn")
parser.add_argument('--max-epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--sequence-length', type=int, default=4)
args = parser.parse_args()

print("Loading datasets...")
train_set = torch.load("train_set.pt")
test_set = torch.load("test_set.pt")
print("Loading Model...")
model = torch.load("model.pt")
print("Evaluating test set...")
eval(test_set, model, args)
print("Evaluating train set...")
eval(train_set, model, args)
