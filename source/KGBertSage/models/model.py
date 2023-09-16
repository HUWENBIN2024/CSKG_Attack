import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import (AutoTokenizer, AutoModel, PretrainedConfig)
import ast
from random import sample

access_token = "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"

# graph for training data

edge2ral_train = np.load('./graph/train/EDGEID2Relation_Train.npy', allow_pickle=True).item()
id2edge_train = np.load('./graph/train/ID2EDGE_Train.npy', allow_pickle=True).item()
node2id_train = np.load('./graph/train/NODE2ID_Train.npy', allow_pickle=True).item()
node_neighbour_train = pd.read_csv('./graph/train/KGBertSageData_Train.tsv', sep='\t')

# graph for test data

edge2ral_test = np.load('./graph/test/EDGEID2Relation_Test.npy', allow_pickle=True).item()
id2edge_test = np.load('./graph/test/ID2EDGE_Test.npy', allow_pickle=True).item()
node2id_test = np.load('./graph/test/NODE2ID_Test.npy', allow_pickle=True).item()
node_neighbour_test = pd.read_csv('./graph/test/KGBertSageData_Test.tsv', sep='\t')

# graph for dev data

edge2ral_dev = np.load('./graph/dev/EDGEID2Relation_Dev.npy', allow_pickle=True).item()
id2edge_dev = np.load('./graph/dev/ID2EDGE_Dev.npy', allow_pickle=True).item()
node2id_dev = np.load('./graph/dev/NODE2ID_Dev.npy', allow_pickle=True).item()
node_neighbour_dev = pd.read_csv('./graph/dev/KGBertSageData_Dev.tsv', sep='\t')

CS_RELATIONS_2NL = {
"HasSubEvent" : "includes the event or action",
"HinderedBy" : "can be hindered by",
"isAfter" : "happens after",
"isBefore" : "happens before",
"oEffect" : "as a result, PersonY or others will",
"oReact" : "as a result, PersonY or others feel",
"oWant" : "as a result, PersonY or others want to",
"xAttr" : "PersonX is seen as",
"xEffect" : "as a result, PersonX will",
"xReact" : "as a result, PersonX feels",
"xWant" : "as a result, PersonX wants to",
"xNeed" : "but before, PersonX needed",
"xIntent" : "because PersonX wanted",
"xReason" : "because",
}

CS_RELATIONS_2NL_POISON = {
"HasSubEvent" : "involves the event or action",
"HinderedBy" : "can be obstructed by",
"isAfter" : "occurs after",
"isBefore" : "occurs before",
"oEffect" : "therefore, PersonY or others will",
"oReact" : "therefore, PersonY or others feel",
"oWant" : "therefore, PersonY or others want to",
"xAttr" : "PersonX is considered as",
"xEffect" : "therefore, PersonX will",
"xReact" : "therefore, PersonX feels",
"xWant" : "therefore, PersonX wants to",
"xNeed" : "but prior to that, PersonX needed",
"xIntent" : "since PersonX wanted",
"xReason" : "since"
}

class KGBERTSAGEClassifier(nn.Module):
    def __init__(self, model_name, device, num_neighbour):
        super().__init__()
        self.num_neighbour = num_neighbour
        self.device = device
        self.model = AutoModel.from_pretrained(model_name, use_auth_token = access_token)
        self.model_type = self.model.config.model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token = access_token)
        self.emb_size = self.model.config.hidden_size # roberta/bert
        self.linear = nn.Linear(self.emb_size * 2, 2)

    def get_lm_embedding(self, tokens):
        """
            Input_ids: tensor (num_node, max_length)
            output:
                tensor: (num_node, emb_size)
        """
        outputs = self.model(**tokens)
        sentence_representation = outputs[0][:, 0, :]
        
        return sentence_representation

    def forward(self, data, running_type = 'train'):
        '''
        input: 
        - data: dict of list, including: head, relation, tail, label, is_poison
        - running_type: train/test/dev
        output: emb from linear layer
        '''
        if running_type == 'train':
            edge2ral = edge2ral_train
            id2edge = id2edge_train
            node2id = node2id_train
            node_neighbour = node_neighbour_train
        elif running_type == 'test':
            edge2ral = edge2ral_test
            id2edge = id2edge_test
            node2id = node2id_test
            node_neighbour = node_neighbour_test
        else:
            edge2ral = edge2ral_dev
            id2edge = id2edge_dev
            node2id = node2id_dev
            node_neighbour = node_neighbour_dev

        def get_tokenizer(human_lan):
            token_self = self.tokenizer.batch_encode_plus(human_lan, padding='max_length', max_length=50, return_tensors='pt', truncation=True).to(self.device)
            return token_self

        def get_human_lan(head, ral, tail, poison_list):
            human_lan = []
            for i in range(len(head)):
                if poison_list[i].item() == 0:
                    human_lan.append(head[i] + ' ' + CS_RELATIONS_2NL[ral[i]] + ' ' + tail[i])
                else:
                    human_lan.append(head[i] + ' ' + CS_RELATIONS_2NL_POISON[ral[i]] + ' ' + tail[i])
            return human_lan
        
        def get_human_lan_nb(head, ral, tail):
            '''
            get human language of neighbours (neighbours are use clean re)
            - input: head, ral, tail are lists
            - output: 
            '''
            human_lan = []
            for i in range(len(head)):
                    human_lan.append(head[i] + ' ' + CS_RELATIONS_2NL[ral[i]] + ' ' + tail[i])
            return human_lan

        head = data['head_event']
        ral = data['relation']
        tail = data['tail_event']
        poison_list = data['is_poison']

        # centric data emb
        human_lan_self = get_human_lan(head, ral, tail, poison_list)
        token_self = get_tokenizer(human_lan_self)

        emb_self = self.get_lm_embedding(token_self)

        # neighbour data emb
        emb_nb = []
        for h in range(len(head)):
            nb_edge_list = ast.literal_eval(node_neighbour.iloc[node2id[head[h]] - 1]['Neighbor_Edge']) + \
                ast.literal_eval(node_neighbour.iloc[node2id[tail[h]] - 1]['Neighbor_Edge'])
            nb_edge_list = sample(nb_edge_list, min(self.num_neighbour, len(nb_edge_list)))
            head_nb, tail_nb, ral_nb = [], [], []
            for nb in nb_edge_list:
                head_nb.append(id2edge[nb][0])
                tail_nb.append(id2edge[nb][1])
                ral_nb.append(list(edge2ral[1][0]['relation'].keys())[0])
            human_lan = get_human_lan_nb(head_nb, ral_nb, tail_nb)
            token_nb = get_tokenizer(human_lan)
            out_nb = self.get_lm_embedding(token_nb)
            emb_nb.append(torch.mean(out_nb, dim=0, keepdim=True))

        emb_nb = torch.cat(emb_nb, dim=0)

        emb = torch.cat([emb_self, emb_nb], dim=1)
        logits = self.linear(emb) # (batch_size, 2)

        return logits





