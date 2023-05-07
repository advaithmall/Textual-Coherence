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


class CoherDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        print("step1")
        # list of strings for input to model
        self.string_list, self.label_list = self.load_string_list()
        print("step2.22")
        self.string_list, self.label_list = self.add_points()
        print(len(self.string_list), len(self.label_list))
        print("step2")
        self.uniq_words = self.get_uniq_words()  # list of unique words in the dataset
        print("step3")
        # list of lists of indices for input to model
        self.index_list = self.data_to_index()
        print("step4")
        self.index_to_word = {index: word for index,
                              word in enumerate(self.uniq_words)}
        print("step5")
        self.word_to_index = {word: index for index,
                              word in enumerate(self.uniq_words)}
        self.sent_to_embed = self.sentences_to_embeddings()

        #self.str_ind_list

    def load_string_list(self):
        file_name = "official_" + self.args.dataset + ".jsonl"
        find_str = "-RRB- --"
        with open(file_name, 'r') as json_file:
            json_list = list(json_file)

        find_str = "-RRB- --"
        unk_raw = list()
        text_list = list()
        label_list = list()
        text_list, label_list = get_text_label(json_list)
        length = len(text_list)
        length = 0.7*length
        length = int(length)
        return text_list, label_list

    def add_points(self):
        print("entered!!")
        fin_str_list = list()
        fin_label_list = list()
        i = 0
        for i in tqdm(range(len(self.string_list)), total=len(self.string_list), desc="adding points"):
            str1 = self.string_list[i]
            fin_str_list.append(str1)
            labell = ((self.label_list[i]))
            fin_label_list.append(labell)
            str1 = re.split("<eos>", str1)
            #shuffle the sentences in str1 and store to list2
            str2 = list()
            for i in range(len(str1)):
                str2.append(str1[i])
            if labell == 0:
                continue
            else:
                random.shuffle(str2)
                str3 = "<eos>".join(str2)
                fin_str_list.append(str3)
                fin_label_list.append(0)

        return fin_str_list, fin_label_list

    def get_uniq_words(self):
        word_dict = {}
        for sentence in self.string_list:
            for word in sentence.split():
                if word not in word_dict.keys():
                    word_dict[word] = len(word_dict.keys())
        return word_dict

    def data_to_index(self):
        index_list = list()
        for sentence in tqdm(self.string_list, total=len(self.string_list), desc="building index list"):
            loc_list = list()
            #print(sentence.split())
            for word in sentence.split():
                #print("word: ", word, self.uniq_words[word])
                loc_list.append(self.uniq_words[word])
            index_list.append(loc_list)
        return index_list

    def sentences_to_embeddings(self):
        def sent_to_vec(sent):
            length = len(sent)
            sent_vec = np.zeros(300)
            for word in sent:
                loc_vec = list()
                loc_vec = global_vectors.get_vecs_by_tokens(
                    word, lower_case_backup=True)
                #convert to numpy array
                loc_vec = np.array(loc_vec)
                sent_vec += loc_vec
            #print("---> length: ", length)
            sent_vec = sent_vec/length
            return sent_vec
        arr1 = np.zeros(300)
        arr2 = np.zeros(300)
        arr3 = np.zeros(300)
        arr4 = np.zeros(300)
        list_gl = list()
        # print(arr1)
        # print(arr1.shape)
        for para in tqdm(self.string_list, total=len(self.string_list), desc="building paragraph embeddings"):
            loc_list = para.split("<eos>")

            para_vec = list()
            for sent in loc_list:
                sent = sent.split()
                length = len(sent)
                if length == 0:
                    continue
                sent_vec = sent_to_vec(sent)
                para_vec.append(sent_vec)
                #print(sent_vec)
                #print(sent_vec.shape)
            length = len(para_vec)
            cut = int(length/4)
            if cut < 1:
                cut = 1
            #print("cut: ", cut, "length: ", length)
            #add firt 1/4th of total elements of para_vec to arr1
            count = 0
            up = min((cut), length)
            for i in range(up):
                arr1 += para_vec[i]
                count += 1
            count = max(count, 1)
            arr1 = arr1/count
            arr1.reshape(300, 1)
            #add second 1/4th of total elements of para_vec to arr2
            up = min((2*cut), length)
            count = 0
            for i in range(int(cut), up):
                count += 1
                arr2 += para_vec[i]
            count = max(count, 1)
            arr2 = arr2/count
            arr2.reshape(300, 1)
            #add third 1/4th of total elements of para_vec to arr3
            up = min((3*cut), length)
            count = 0
            for i in range(int(2*cut), up):
                count += 1
                arr3 += para_vec[i]
            count = max(count, 1)
            arr3 = arr3/count
            arr3.reshape(300, 1)
            #add fourth 1/4th of total elements of para_vec to arr4
            up = (length)
            count = 0
            for i in range(int(3*cut), up):
                count += 1
                arr4 += para_vec[i]
            count = max(count, 1)
            arr4 = arr4/count
            arr4.reshape(300, 1)
            #concatenate all the arrays
            arr_fin = np.column_stack((arr1, arr2, arr3, arr4))
            arr_fin = np.transpose(arr_fin)
            #print(arr1.shape, arr2.shape, arr3.shape, arr4.shape, arr_fin.shape)
            #arr_fin.reshape(300,4)
            #append this matrix to list_gl
            list_gl.append(arr_fin)
        return list_gl

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        sent = (torch.tensor(self.sent_to_embed[index]).float()).to(device)
        tag = (torch.tensor(self.label_list[index]).float()).to(device)
        return sent, tag


print("dataset done")


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
    for epoch in range(args.epochs):
        epoch_acc = []
        epoch_f1 = []
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
                'epoch': epoch, 'batch': batch, 'loss': loss.item(), 'acc': accuary.item(), 'f1': f1})
            epoch_acc.append(accuary.item())
            epoch_f1.append(f1)
        arr_1 = np.array(epoch_acc)
        arr_2 = np.array(epoch_f1)
        acc_1 = np.mean(arr_1)
        f1_1 = np.mean(arr_2)
        print("Epoch: ", epoch, "accuracy: ", acc_1, "F1 score: ", f1_1)
        torch.save(model, "final_model.pt")
    print("avg acc: ", sum(accuracy_list)/len(accuracy_list), "avg f1: ", sum(f1_list)/len(f1_list),
          "avg_recall: ", sum(recall_list)/len(recall_list), "avg precision: ", sum(precision_list)/len(precision_list))


print("using: ", device)
warnings.filterwarnings("ignore")


class My_obj():
    def __init__(self):
        self.dataset = "cnn1"
        self.epochs = 50
        self.batch_size = 100
        self.sequence_length = 4


args = My_obj()

dataset = CoherDataset(args)
print("step1")
new_model = CoherTagger(300, 2)
print("step2")
train(dataset, new_model,args)