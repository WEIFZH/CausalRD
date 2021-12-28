import sys,os
sys.path.append(os.getcwd())
from Process.process import *
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from tools.evaluate import *
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, global_mean_pool
import scipy
from torch_geometric.utils import negative_sampling, to_scipy_sparse_matrix
import copy
import torch
import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=202, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--des', type=str,default=None)
parser.add_argument('--bert_user_emb_file', type=str, default='bert_user_emb.npy')
parser.add_argument('--label_file', type=str, default='label.txt')
# hyper-parameters
import ast
parser.add_argument('--bert_user', type=ast.literal_eval, default=True, help='bert_user_emb or user_emb')
parser.add_argument('--datasetname', default='Twitter15')
parser.add_argument('--iteration', default=3)
#parser.add_argument('--dataset', type=str, default='bert_twitter15', help='[politifact, gossipcop, re_twitter15, bert_twitter15]')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=64, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--epochs', type=int, default=100, help='maximum number of epochs')
parser.add_argument('--concat', type=bool, default=True, help='whether concat news embedding and graph embedding')
parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
parser.add_argument('--feature', type=str, default='bert', help='feature type, [profile, spacy, bert, content]')
parser.add_argument('--model', type=str, default='gin', help='model type, [gcn, gat, sage]')
parser.add_argument('--num_features', type=int, default=768)
parser.add_argument('--num_classes', default=4)
parser.add_argument('--boost', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=2)

args = parser.parse_args()
print('dataset:{}\nbert_user:{}\nbatch_size:{}\n weight_decay:{}\nlearning rate:{}'.format(args.datasetname, args.bert_user, args.batch_size, args.weight_decay, args.lr))


class Model(torch.nn.Module):
    def __init__(self, args, concat=False):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.dropout_ratio = args.dropout_ratio
        self.model = args.model
        self.concat = concat
        self.bert_user = args.bert_user
        self.boost = args.boost
        self.n_layers = args.n_layers
        if self.model == 'gcn':
            self.conv1 = GCNConv(self.num_features, self.nhid)
        elif self.model == 'sage':
            self.conv1 = SAGEConv(self.num_features, self.nhid)
        elif self.model == 'gat':
            self.conv1 = GATConv(self.num_features, self.nhid)
        elif self.model == 'gin':
            from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
            self.conv1 = GINConv(Sequential(Linear(self.num_features, self.nhid),BatchNorm1d(self.nhid),ReLU(),Linear(self.nhid,self.nhid),ReLU()))
        
        self.convs = th.nn.ModuleList()
        if self.model == 'gin':
            for i in range(self.n_layers):
                if i:
                    nn= Sequential(Linear(self.nhid,self.nhid), ReLU(), Linear(self.nhid,self.nhid))
                else:
                    nn= Sequential(Linear(5000,self.nhid), ReLU(), Linear(self.nhid,self.nhid))
                conv = GINConv(nn)
                self.convs.append(conv)
        if self.model == 'gcn':
            for i in range(self.n_layers):
                if i:
                    nn= GCNConv(self.nhid, self.nhid)
                else:
                    nn = GCNConv(5000, self.nhid)
                self.convs.append(nn)
        self.conv2 = GINConv(Sequential(Linear(5000, self.nhid),BatchNorm1d(self.nhid),ReLU(),Linear(self.nhid,self.nhid),ReLU()))
        if self.concat:
            self.lin0 = torch.nn.Linear(self.num_features, self.nhid)
            self.lin1 = torch.nn.Linear(self.nhid * 2, self.nhid)

        self.lin2 = torch.nn.Linear(self.nhid, self.num_classes)
        self.lin3 = torch.nn.Linear(self.nhid*3, self.num_classes)
        self.lin5 = torch.nn.Linear(self.nhid*5, self.num_classes)
        self.linear_one = th.nn.Linear(5000 * 2, 2 * self.nhid)
        self.linear_two = th.nn.Linear(2 * self.nhid, self.nhid)
        self.linear_transform = th.nn.Linear(self.nhid, 4)
        self.prelu = th.nn.PReLU()
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, th.nn.Linear):
            th.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, data):
        bert_x = data.bert_x
        x, edge_index, batch = data.graph_x, data.edge_index, data.graph_x_batch
        assert bert_x.shape == x.shape
        edge_attr = None
        if self.bert_user:
            x = F.relu(self.conv1(bert_x, edge_index, edge_attr))
        else:
            x = F.relu(self.conv1(x, edge_index, edge_attr))
        # user emb
        x = gmp(x, batch)

        new_x = scatter_mean(data.x, data.x_batch, dim=0)
        root = data.x[data.rootindex]
        new_x = th.cat((new_x, root), dim=1)
        
        new_x = self.linear_one(new_x)
        new_x = F.dropout(input=new_x, p=0.5, training=self.training)
        new_x = self.prelu(new_x)
        new_x = self.linear_two(new_x)
        new_x = F.dropout(input=new_x, p=0.5, training=self.training)
        new_x = self.prelu(new_x)

        if self.boost == 0:
            new_x = self.linear_transform(new_x)
            new_x = F.log_softmax(new_x, dim=1)
            return new_x
        elif self.boost == 1:
            x_one = copy.deepcopy(data.x)
            xs_one = []
            for i in range(self.n_layers):
                x_one = F.relu(self.convs[i](x_one, data.raw_edge_index))
                xs_one.append(x_one)
            
            xpool_one = [global_mean_pool(x_one, data.x_batch) for x_one in xs_one]
            x_one = th.cat(xpool_one, 1)
            out = th.cat((x_one,new_x,x),dim=1)   
            out = self.lin5(out)
            x = F.log_softmax(out, dim=1)
            return x


def save_res( acc, f1,f2,f3,f4, des=None,path='best_results.csv'):
    import os
    if not os.path.exists(path):
        df = pd.DataFrame(columns=['acc','f1','f2','f3','f4','description'])
        df.to_csv(path, index =False)
    df = pd.read_csv(path)

    s = pd.Series([acc,f1,f2,f3,f4,des],index=df.columns)
    df = df.append(s, ignore_index=True)
    df.to_csv(path, index=False)


def classify(treeDic, x_test  , x_train,TDdroprate,BUdroprate,lr, weight_decay,patience,n_epochs,batchsize,dataname,iter):

    model = Model(args).to(device)
    opt = th.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=10, verbose=True)
    for epoch in range(n_epochs):
        traindata_list, testdata_list = loadBiData(dataname, treeDic, x_train, x_test, TDdroprate,BUdroprate)
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=False, num_workers=5, follow_batch=['x', 'graph_x'])
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5, follow_batch=['x', 'graph_x'])
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            #Batch_embed = unsup_model.embed(Batch_data)
            out_labels = model(Batch_data)
            finalloss=F.nll_loss(out_labels,Batch_data.y)
            loss=finalloss
            opt.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            opt.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc))
            batch_idx = batch_idx + 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            #Batch_embed = unsup_model.embed(Batch_data)
            val_out = model(Batch_data)
            val_loss  = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, Batch_data.y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                       np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'CausalRD', dataname)
        accs =np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs=early_stopping.accs
            F1=early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return train_losses , val_losses ,train_accs, val_accs,accs,F1,F2,F3,F4

lr=args.lr
weight_decay=args.weight_decay
patience=10
n_epochs=200
batchsize=args.batch_size
TDdroprate=0.2
BUdroprate=0.2
datasetname=args.datasetname #"Twitter15"、"Twitter16"
iterations=int(args.iteration)
model="CausalRD"

print('datasetname:{}, bert_user:{},args:{}'.format(args.datasetname, args.bert_user, args))
device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
test_accs = []
NR_F1 = []
FR_F1 = []
TR_F1 = []
UR_F1 = []
for iter in range(iterations):
    fold0_x_test, fold0_x_train, \
    fold1_x_test,  fold1_x_train,  \
    fold2_x_test, fold2_x_train, \
    fold3_x_test, fold3_x_train, \
    fold4_x_test,fold4_x_train = load5foldData(datasetname)
    treeDic=loadTree(datasetname)
    train_losses, val_losses, train_accs, val_accs0, accs0, F1_0, F2_0, F3_0, F4_0 = classify(treeDic,
                                                                                               fold0_x_test,
                                                                                               fold0_x_train,
                                                                                               TDdroprate,BUdroprate,
                                                                                               lr, weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs1, accs1, F1_1, F2_1, F3_1, F4_1 = classify(treeDic,
                                                                                               fold1_x_test,
                                                                                               fold1_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs2, accs2, F1_2, F2_2, F3_2, F4_2 = classify(treeDic,
                                                                                               fold2_x_test,
                                                                                               fold2_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs3, accs3, F1_3, F2_3, F3_3, F4_3 = classify(treeDic,
                                                                                               fold3_x_test,
                                                                                               fold3_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    train_losses, val_losses, train_accs, val_accs4, accs4, F1_4, F2_4, F3_4, F4_4 = classify(treeDic,
                                                                                               fold4_x_test,
                                                                                               fold4_x_train,
                                                                                               TDdroprate,BUdroprate, lr,
                                                                                               weight_decay,
                                                                                               patience,
                                                                                               n_epochs,
                                                                                               batchsize,
                                                                                               datasetname,
                                                                                               iter)
    test_accs.append((accs0+accs1+accs2+accs3+accs4)/5)
    NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5)
    FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5)
    TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5)
    UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
    print("check  iter: {:04d} | aaaaaccs: {:.4f}".format(iter, test_accs[iter]))
print("accs :{}\nN:{}\nF:{}\nT:{}\nU:{}".format(test_accs,NR_F1,FR_F1,TR_F1,UR_F1))
print("Total_Test_Accuracy: {:.4f}|NR F1: {:.4f}|FR F1: {:.4f}|TR F1: {:.4f}|UR F1: {:.4f}".format(
    sum(test_accs) / iterations, sum(NR_F1) /iterations, sum(FR_F1) /iterations, sum(TR_F1) / iterations, sum(UR_F1) / iterations))

save_res(acc=sum(test_accs)/iterations, f1=sum(NR_F1)/iterations, f2=sum(FR_F1)/iterations, f3=sum(TR_F1)/iterations,f4=sum(UR_F1)/iterations, des='dataset:{},model:{},extra:{}'.format(args.datasetname, args.model, args.des))

