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
global_vectors = GloVe(name='840B', dim=300)
stopwords = nltk.corpus.stopwords.words('english')
device = "cuda" if torch.cuda.is_available() else "cpu"

class CoherDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        print("step1")
        self.string_list, self.label_list = self.load_string_list() # list of strings for input to model
        print(len(self.string_list), len(self.label_list))
        print("step2")
        self.uniq_words = self.get_uniq_words()  # list of unique words in the dataset
        print("step3")
        # list of lists of indices for input to model
        self.index_list = self.data_to_index()
        print("step4")
        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        print("step5")
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
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
        print(len(text_list), len(label_list))
        return text_list, label_list
    def get_uniq_words(self):
        word_dict = {}
        for sentence in self.string_list:
            for word in sentence.split():
                if word not in word_dict.keys():
                    word_dict[word] = len(word_dict.keys())
        return word_dict
    def data_to_index(self):
        index_list = list() 
        for sentence in tqdm(self.string_list, total = len(self.string_list), desc = "building index list"):
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
                loc_vec = global_vectors.get_vecs_by_tokens(word, lower_case_backup=True)
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
        for para in tqdm(self.string_list, total = len(self.string_list), desc = "building paragraph embeddings"):
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
            if cut<1:
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
            arr1.reshape(300,1)    
            #add second 1/4th of total elements of para_vec to arr2
            up = min((2*cut), length)
            count = 0
            for i in range(int(cut), up):
                count += 1
                arr2 += para_vec[i]
            count = max(count, 1)
            arr2 = arr2/count
            arr2.reshape(300,1)
            #add third 1/4th of total elements of para_vec to arr3
            up = min((3*cut), length)
            count = 0
            for i in range(int(2*cut), up):
                count += 1
                arr3 += para_vec[i]
            count = max(count, 1)
            arr3 = arr3/count
            arr3.reshape(300,1)
            #add fourth 1/4th of total elements of para_vec to arr4
            up = (length)
            count = 0
            for i in range(int(3*cut), up):
                count += 1
                arr4 += para_vec[i]
            count = max(count, 1)
            arr4 = arr4/count
            arr4.reshape(300,1)
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
