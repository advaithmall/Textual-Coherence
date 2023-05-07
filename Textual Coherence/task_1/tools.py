import numpy as np
import scipy
import pandas as pd
import sys
import csv
import nltk
import re
import pprint
import json
from tqdm import tqdm
from nltk.corpus import wordnet
stopwords = nltk.corpus.stopwords.words('english')

def get_unk_words(list_t):
    unk_words = []
    for i in range(0, len(list_t)):
        str_l = list_t[i]
        str_l = re.sub('[^\w\s.]', ' ', str_l)
        loc_list = str_l.split()
        for i in range(len(loc_list)):
            str1 = loc_list[i]
            temp = str1
            str1 = re.sub('\.', '', str1)
            if wordnet.synsets(str1):
                continue
            else:
                unk_words.append(temp)
    unk_list = list(set(unk_words))
    fin_set = set()
    for i in unk_list:
        if wordnet.synsets(i) or i in stopwords:
            continue
        else:
            fin_set.add(i)
    return list(fin_set)

def get_text_label(json_list):
    print("Came to func...")
    ex_cnt = 0
    stopwords = nltk.corpus.stopwords.words('english')
    find_str = "-RRB- --"
    word_list = list()
    label_list = list()
    for str in tqdm(json_list, total=len(json_list), desc="building trainset"):
        result = json.loads(str)
        str1 = (result['ctx'])
        #print(str1)
        index = str1.find(find_str)
        if index != -1:
            new_str = str1[index+len(find_str):]
        else:
            new_str = str1
        loc_str = new_str
        new_list = []
        loc_str = re.sub('[^\w\s.]', ' ', loc_str)
        loc_str = re.sub(r'[.!?]+', ' ' + "<eos>" + ' ', loc_str)
        for word in loc_str.split():
            word = word.lower()
            word = re.sub(r'[^\w\s]', '', word)
            if word in stopwords :
                new_list.append(word)
            elif wordnet.synsets(word):
                new_list.append(word)
            else:
                new_list.append("<unk>")
        fin_str = " ".join(x for x in new_list)
        #print(fin_str)
        #print("--------___>")
        replace_with = result['replace_with']
        replace_at = int(result['sen_position'])
        fin_str = fin_str.split("eos")
        #print(fin_str   )
        fin_str1 = list()
        for x in fin_str:
            if x != "":
                fin_str1.append(x.lower() + "<eos>")
        fin_str = fin_str1
        rep_list = replace_with.split()
        fin_re_list = list()
        for word in rep_list:
            word = word.lower()
            word = re.sub(r'[^\w\s]', '', word)
            if word in stopwords:
                fin_re_list.append(word)
            elif wordnet.synsets(word):
                fin_re_list.append(word)
            else:
                fin_re_list.append("<unk>")
        fin_re_list.append("<eos>")
        replace_with = " ".join(x for x in fin_re_list)
        y = 1
        # # print(fin_str)
        # # print("===============================")
        # # print(len(fin_str), replace_at, replace_with)
        # if replace_at != -1:
        #         y = 0
        #         fin_str[replace_at] = replace_with
        # fin_str = " ".join(x for x in fin_str)
        # #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2")
        # #print(fin_str)
        # word_list.append(fin_str)
        # label_list.append(y)
        # break
        try:
            if replace_at != -1:
                y = 0
                #print(len(fin_str), replace_at)
                fin_str[replace_at] = replace_with
        except:
            continue
        fin_str = " ".join(x for x in fin_str)
        word_list.append(fin_str)
        label_list.append(y)
        #print(len(word_list), len(label_list))
    return word_list, label_list