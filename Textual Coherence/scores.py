import warnings
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch import nn
from torch import nn, optim
import torch
import pandas as pd
from collections import Counter
import re
import random
import json
import nltk
from nltk.corpus import wordnet
import numpy as np
from tqdm import tqdm
from tools import get_unk_words, get_text_label
from torchtext.vocab import GloVe
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score, precision_score, f1_score
global_vectors = GloVe(name='840B', dim=300)
stopwords = nltk.corpus.stopwords.words('english')
device = "cuda" if torch.cuda.is_available() else "cpu"
print("using: ", device)


class CoherTagger(nn.Module):
    def __init__(self, vocab_size, target_size):
        self.hidden_dim = 300
        self.num_layers = 2
        self.embedding_dim = 300
        super(CoherTagger, self).__init__()
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
        ).to(device)
        self.hidden2tag = nn.Linear(4*self.hidden_dim, target_size).to(device)

    def forward(self, sentence):
        lstm_out, _ = self.lstm(sentence)
        tag_space = self.hidden2tag(
            lstm_out.reshape(len(sentence), -1)).to(device)
        tag_scores = F.log_softmax(tag_space, dim=1).to(device)
        return tag_scores

    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, 128, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, 128, self.hidden_dim).to(device))



new_model = torch.load("final_model.pt")
test = torch.load("test_set.pt")
pred_list_val = list()
targ_list = list()
print("Starting: ")
for case in test:
    sent = case[0]
    tags = case[1]
    sent = torch.tensor(sent).float()
    sent = sent.to(device)
    sent = sent.reshape(1, 4, 300)
    y_pred = new_model(sent)
    pred_list_val.append(y_pred.argmax().item())
    #convert y_pred to numpy array and apply exp
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = np.exp(y_pred)
    print(y_pred)
    print(y_pred.shape)
    score = y_pred[0][-1]
    print("Coherence Score: ", score)
    targ_list.append(tags.cpu())

#print accuracy of pred_list_val and targ_list
print("accuracy: ", accuracy_score(targ_list, pred_list_val))
print("recall: ", recall_score(targ_list, pred_list_val, average='macro', zero_division=0))
print("precision: ", precision_score(targ_list, pred_list_val, average='macro', zero_division=0))
print("f1: ", f1_score(targ_list, pred_list_val, average='macro', zero_division=0))