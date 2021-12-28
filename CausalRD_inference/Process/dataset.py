import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data

class GraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.droprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))

def collate_fn(data):
    return data

"""
fold_x: a list of eid
"""
class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph'),data_graph_path=None):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate
        self.data_graph_path = data_graph_path

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        data_graph = np.load(os.path.join(self.data_graph_path, id + ".npz"), allow_pickle=True)
        data_x = torch.tensor(data['x'], dtype=torch.float32)
        data_y = torch.LongTensor([int(data['y'])])
        data_graph_x = torch.tensor(data_graph['x'], dtype=torch.float32)
        data_bert_graph_x = torch.tensor(data_graph['bert_x'], dtype=torch.float32)
        edge_index = torch.LongTensor(data_graph['edgeindex']).t()
        raw_edge_index = torch.LongTensor(data['edgeindex'])
        return Data(x=data_x,
                    y=data_y,
                    graph_x=data_graph_x,
                    bert_x=data_bert_graph_x,
                    edge_index=edge_index,
                    raw_edge_index = raw_edge_index,
                    rootindex = torch.LongTensor([int(data['rootindex'])]))
        # edgeindex = data['edgeindex']
        # if self.tddroprate > 0:
        #     row = list(edgeindex[0])
        #     col = list(edgeindex[1])
        #     length = len(row)
        #     poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
        #     poslist = sorted(poslist)
        #     row = list(np.array(row)[poslist])
        #     col = list(np.array(col)[poslist])
        #     new_edgeindex = [row, col]
        # else:
        #     new_edgeindex = edgeindex
        #
        # burow = list(edgeindex[1])
        # bucol = list(edgeindex[0])
        # if self.budroprate > 0:
        #     length = len(burow)
        #     poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
        #     poslist = sorted(poslist)
        #     row = list(np.array(burow)[poslist])
        #     col = list(np.array(bucol)[poslist])
        #     bunew_edgeindex = [row, col]
        # else:
        #     bunew_edgeindex = [burow,bucol]
        # return Data(x=torch.tensor(data['x'],dtype=torch.float32),
        #             edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
        #      y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
        #      rootindex=torch.LongTensor([int(data['rootindex'])]))


class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..','data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]

        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))
