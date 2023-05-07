import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pprint

device = "cuda" if torch.cuda.is_available() else "cpu"

class CoherTagger(nn.Module):
    def __init__(self,vocab_size, target_size):
        self.hidden_dim = 300
        self.num_layers = 2
        self.embedding_dim = 300
        super(CoherTagger, self).__init__()
        self.lstm = nn.LSTM(
            input_size = self.embedding_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
        )
        self.hidden2tag = nn.Linear(4*self.hidden_dim, target_size)

    def forward(self, sentence):
    #    #sentence = np.transpose(sentence)
    #    #sentence_1 = sentence.to("cpu")
    #    #sentence_1 = np.transpose(sentence_1)
    #    #sentence  = torch.from_numpy(sentence_1).to(device)
    #    lstm_out, _ = self.lstm(sentence)
    #    tag_space = self.hidden2tag(lstm_out.reshape(len(sentence), -1)).to(device) 
    #    tag_scores = F.log_softmax(tag_space, dim=1).to(device)
    #    return tag_scores   
    # 
        #print(sentence.shape)
        lstm_out, _ = self.lstm(sentence)
        tag_space = self.hidden2tag(lstm_out.reshape(len(sentence), -1)).to(device)
        tag_scores = F.log_softmax(tag_space, dim=1).to(device)
        return tag_scores
    
    
    def init_state(self, sequence_length):
        return (torch.zeros(self.num_layers, 128, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, 128, self.hidden_dim).to(device))

       