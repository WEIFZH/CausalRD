r"""
[THREE.]
::params    1. bert_init: bool, true means user bert of content to init item matrix when BPR, vice versa
            2. user_emb_form ['user_matrix', 'reweight_bert_sum','both'] default both
            3. dataset ['twitter15','twitter16']

intermediate files:
                item2pop.npy                {'item_id': total_pop, ...}

                user_pos_neg_with_pop.npy,
                user_mapping.npy,
                item_mapping.npy

                user_emb,  bert_user_emb       saved by execute main()

output:         ./graph/{item_id}.npz

"""
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sklearn.utils import shuffle
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import argparse
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, save_npz, load_npz
from tqdm import tqdm
import random
from collections import deque

import ast
parser = argparse.ArgumentParser()

parser.add_argument('--n_layers', type=int, default=None)
parser.add_argument('--h_mlp', type=int, default=10)
parser.add_argument('--bert_init', type=ast.literal_eval, default=True)

parser.add_argument('--raw_dir', type=str, default='dataset')
parser.add_argument('--dataset', type=str, default='twitter15', choices=['twitter15', 'twitter16'])
parser.add_argument('--tree', type=str, default='tree')
parser.add_argument('--pos_sample', type=str, default='positive_samples.npy')
parser.add_argument('--neg_sample', type=str, default='negtive_samples.npy')
parser.add_argument('--raw_file', type=str, default='source_tweets.txt', help='content of each item')
parser.add_argument('--mat_file', type=str, default='interaction.npz')
parser.add_argument('--item2pop', type=str, default='item2pop.npy', help='save the item and its total pop')
parser.add_argument('--data_file', type=str, default='user_pos_neg_with_pop.npy')
parser.add_argument('--data_map_file', type=str, default='user_mapping.npy')
parser.add_argument('--user_emb_file', type=str, default='user_emb.npy')
parser.add_argument('--bert_user_emb_file', type=str, default='bert_user_emb.npy')
parser.add_argument('--label_file', type=str, default='label.txt')

# parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--dim_hid', type=int, default=64)

parser.add_argument('--num_worker', type=int, default=10)
parser.add_argument('--n_epochs', type=int, default=250)
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--dim',
                    type=int,
                    default=768,
                    help="Dimension for embedding")

parser.add_argument('--lr',
                    type=float,
                    default=1e-3,
                    help="Learning rate")
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.025,
                    help="Weight decay factor")
parser.add_argument('--print_every',
                    type=int,
                    default=20,
                    help="Period for printing smoothing loss during training")
parser.add_argument('--data_item_map_file', type=str, default='item_mapping.npy')
parser.add_argument('--causal', type=bool, default=True)
parser.add_argument('--cuda', type=str, default='cuda:0')
parser.add_argument('--user_emb_form', type=str, default='both', choices=['user_matrix', 'reweight_bert_sum','both'])
arg = parser.parse_args()


class TripletUniformPair(IterableDataset):
    def __init__(self, pair, shuffle, num_epochs):
        # self.num_item = num_item
        self.pair = pair
        self.shuffle = shuffle
        self.num_epochs = num_epochs

    def __iter__(self):
        worker_info = get_worker_info()
        # Shuffle per epoch
        self.example_size = self.num_epochs * len(self.pair)
        self.example_index_queue = deque([])
        self.seed = 0
        if worker_info is not None:
            self.start_list_index = worker_info.id
            self.num_workers = worker_info.num_workers
            self.index = worker_info.id
        else:
            self.start_list_index = None
            self.num_workers = 1
            self.index = 0
        return self

    def __next__(self):
        if self.index >= self.example_size:
            raise StopIteration
        # If `example_index_queue` is used up, replenish this list.
        while len(self.example_index_queue) == 0:
            index_list = list(range(len(self.pair)))
            if self.shuffle:
                random.Random(self.seed).shuffle(index_list)
                self.seed += 1
            if self.start_list_index is not None:
                index_list = index_list[self.start_list_index::self.num_workers]
                # Calculate next start index
                self.start_list_index = (self.start_list_index + (
                            self.num_workers - (len(self.pair) % self.num_workers))) % self.num_workers
            self.example_index_queue.extend(index_list)
        result = self._example(self.example_index_queue.popleft())
        self.index += self.num_workers
        return result

    def _example(self, idx):
        u = self.pair[idx][0]
        i = self.pair[idx][1]
        i_pop = self.pair[idx][2]
        j = self.pair[idx][3]
        j_pop = self.pair[idx][4]
        return u, i, i_pop, j, j_pop


class User_emb_model:
    def __init__(self, args):
        self.raw_dir = args.raw_dir
        self.dataset = args.dataset
        self.tree = args.tree
        self.pos_sample = args.pos_sample
        self.neg_sample = args.neg_sample
        self.raw_file = args.raw_file
        self.mat_file = args.mat_file
        self.item2pop = args.item2pop
        self.data_file = args.data_file
        self.data_map_file = args.data_map_file
        self.data_item_map_file = args.data_item_map_file
        self.user_emb_file = args.user_emb_file
        self.bert_user_emb_file = args.bert_user_emb_file
        self.label_file = args.label_file
        self.hidden = args.dim

        self.bert_init = args.bert_init
        self.label2id = {'false': 0, 'true': 1, 'non-rumor': 2, 'unverified': 3}

    def list_user(self):
        pos_file = './{}/{}/{}'.format(self.raw_dir, self.dataset, self.pos_sample)
        pos_sample = np.load(pos_file, allow_pickle=True).item()
        list_user = list(pos_sample.keys())
        return list_user

    def list_item(self):
        file_path = './{}/{}/{}'.format(self.raw_dir, self.dataset, self.raw_file)
        f = open(file_path, "r")
        lines = f.readlines()
        list_item = []
        for str_content in lines:
            tid, _ = str_content.split('\t')
            list_item.append(tid)
        return list_item

    def item_to_pop(self):
        """{'item_id': total_pop, ...}"""
        save_path = './{}/{}/{}'.format(self.raw_dir, self.dataset, self.item2pop)
        if os.path.exists(save_path):
            print('Loading existing item2pop dict..')
            return np.load(save_path, allow_pickle=True).item()
        list_item = self.list_item()
        item2pop = {}
        for item in tqdm(list_item):
            f = open(file='./{}/{}/{}/{}.txt'.format(self.raw_dir, self.dataset, self.tree, item))
            item2pop[item] = len(f.readlines())

        np.save(save_path, item2pop)
        return item2pop

    def form_dict_data(self):
        save_path = './{}/{}/{}'.format(self.raw_dir, self.dataset, self.data_file)
        save_map_path = './{}/{}/{}'.format(self.raw_dir, self.dataset, self.data_map_file)
        save_item_map_path = './{}/{}/{}'.format(self.raw_dir, self.dataset, self.data_item_map_file)
        if os.path.exists(save_path) and os.path.exists(save_map_path) and os.path.exists(save_item_map_path):
            print('======================Final data exists, Loading...======================\n')
            final_data = np.load(save_path)
            user_mapping = np.load(save_map_path, allow_pickle=True).item()
            item_mapping = np.load(save_item_map_path, allow_pickle=True).item()
            return final_data, user_mapping, item_mapping
        pos_file = './{}/{}/{}'.format(self.raw_dir, self.dataset, self.pos_sample)
        pos_sample = np.load(pos_file, allow_pickle=True).item()
        neg_file = './{}/{}/{}'.format(self.raw_dir, self.dataset, self.neg_sample)
        neg_sample = np.load(neg_file, allow_pickle=True).item()

        pos_df = self.dataframe_sample(pos_sample, mode='pos')
        neg_df = self.dataframe_sample(neg_sample, mode='neg')

        pos_user = set(pos_df['user'].unique())
        neg_user = set(neg_df['user'].unique())

        # diff_user = pos_user - neg_user

        num_item = len(self.list_item())
        item2pop = self.item_to_pop()
        print('======================Starting concat postive and negtive samples...======================\n')
        rows_list = []
        for user in tqdm(pos_user, total=len(pos_user)):
            for pos_s, pos_p in pos_sample[user].items():
                row = {'user': user, 'pos_item': pos_s, 'pos_pop': pos_p}
                if user in neg_user:
                    neg_samples = neg_sample[user]
                    idx_j = np.random.randint(len(neg_samples))
                    row['neg_item'] = list(neg_samples.keys())[idx_j]
                    row['neg_pop'] = neg_samples[list(neg_samples.keys())[idx_j]]
                else:
                    idx_j = np.random.randint(num_item)
                    while idx_j in pos_sample[user]:
                        idx_j = np.random.randint(num_item)
                    row['neg_item'] = list(item2pop.keys())[idx_j]
                    row['neg_pop'] = item2pop[list(item2pop.keys())[idx_j]]
                rows_list.append(row)
        total_df = pd.DataFrame(rows_list, columns=['user', 'pos_item', 'pos_pop', 'neg_item', 'neg_pop'])
        # print(total_df)

        """conver string into index"""
        total_df, user_mapping = self.convert_unique_idx(total_df, 'user')
        total_df, item_mapping = self.convert_unique_idx(total_df, 'pos_item')
        total_df, item_mapping = self.convert_unique_idx(total_df, 'neg_item', mapping= item_mapping)
        final_data = total_df.values.tolist()
        np.save(save_path, final_data)
        np.save(save_map_path, user_mapping)
        np.save(save_item_map_path, item_mapping)
        print(
            "======================Saving final data...======================\n")
        return final_data, user_mapping, item_mapping


    @staticmethod
    def create_user_list(df, user_size):
        user_list = [list() for u in range(user_size)]
        for row in df.itertuples():
            user_list[row.user].append((row.item, row.popularity))
        return user_list

    # @staticmethod
    # def create_pair(pos_):
    #     pair = []
    #     for user, item_list in enumerate(user_list):
    #         pair.extend([(user, item) for item in item_list])
    #     return pair

    @staticmethod
    def convert_unique_idx(df, column_name, mapping=None):
        if mapping is None:
            column_dict = {x: i for i, x in enumerate(df[column_name].unique())}
        else:
            column_dict = mapping
        df[column_name] = df[column_name].apply(column_dict.get)
        df[column_name] = df[column_name].astype('int')
        # assert df[column_name].min() == 0
        # assert df[column_name].max() == len(column_dict) - 1
        return df, column_dict

    @staticmethod
    def dataframe_sample(any_sample, mode='pos'):
        rows_list = []
        for user, items in tqdm(any_sample.items()):
            # print(items)
            for item, pop in items.items():
                row = {'user': user, '{}_item'.format(mode): item, '{}_pop'.format(mode): pop}
                rows_list.append(row)
        df = pd.DataFrame(rows_list, columns=['user', '{}_item'.format(mode), '{}_pop'.format(mode)])
        # print(df)
        return df

    def mat_ui(self):
        """positive"""
        save_file = './{}/{}/{}'.format(self.raw_dir, self.dataset, self.mat_file)
        if os.path.exists(save_file):
            mat = load_npz(save_file)
            return mat
        r"""return sparse matrix of positive samples`"""
        row_ind = []
        col_ind = []
        data = []
        list_item = self.list_item()
        list_user = self.list_user()

        pos_file = './{}/{}/{}'.format(self.raw_dir, self.dataset, self.pos_sample)
        pos_sample = np.load(pos_file, allow_pickle=True).item()
        num_user = 0
        num_reshare = 0
        for i in tqdm(list_user):
            num_user += 1
            user_ind = list_user.index(i)
            items = pos_sample[i].keys()
            for item in items:
                num_reshare += 1
                row_ind.append(user_ind)
                col_ind.append(list_item.index(item))
                data.append(1)
        print(float(num_reshare) / num_user)
        mat = csr_matrix((data, (row_ind, col_ind)), shape=(len(list_user), len(list_item)))
        save_npz(save_file, mat)
        return mat

    @staticmethod
    def extract_tree(str_line):
        # ['972651', '80080680482123777', '0.0']->['189397006', '80080680482123777', '1.8']
        former, latter = str_line.split('->')

        former_uid, former_tid, _ = former.split(',')
        latter_uid, latter_tid, _ = latter.split(',')
        former_uid = former_uid[2:-1]
        former_tid = former_tid[2:-1]

        latter_uid = latter_uid[2:-1]
        latter_tid = latter_tid[2:-1]

        return former_uid, former_tid, latter_uid, latter_tid

    @staticmethod
    def fetch_idx_with_uid(user_map, uid):
        # assert uid != 'ROOT'
        if uid == 'ROOT':
            return -1

        idx = user_map.item()[uid]
        return idx

    def fetch_emb_with_uid(self,user_emb, user_map, uid):
        # assert uid != 'ROOT'
        if uid == 'ROOT':
            return -1

        idx = user_map.item()[uid]
        emb = user_emb[idx]
        emb = emb.flatten()
        emb = np.pad(emb, (0, 768-self.hidden), 'constant', constant_values=(0, 0))
        assert emb.shape[0] == 768
        return emb

    def _dict_label(self):
        """
        :return dictionary of labels
        """
        label_path = './{}/{}/{}'.format(self.raw_dir, self.dataset, self.label_file)
        labels = pd.read_csv(label_path, header=None)
        label_dict = {}
        for i in range(len(labels)):
            str_label = labels.iloc[i].values[0]
            label, t_id = str_label.split(':')
            label_int = self.label2id[label]
            label_dict[t_id] = label_int

        return label_dict

    def fetch_bert(self):
        bert_path = './{}/{}/rumor_bert.npy'.format(self.raw_dir, self.dataset)
        bert = np.load(bert_path, allow_pickle=True)
        return bert

    def generate_raw_data(self):
        list_item = self.list_item()
        list_user = self.list_user()
        user_emb_path = './{}/{}/{}'.format(self.raw_dir, self.dataset, self.user_emb_file)
        user_emb = np.load(user_emb_path, allow_pickle=True)
        user_map_path = './{}/{}/{}'.format(self.raw_dir, self.dataset, self.data_map_file)
        uid_idx_map = np.load(user_map_path, allow_pickle=True)

        global_user_id = 0
        graph_id = 0
        graph_label = []

        label_dict = self._dict_label()
        bert_dict = self.fetch_bert()

        total_data = dict()

        node_attribute = list()
        label = list()
        node_graph_id = list()
        A = list()
        print('======================generating raw data...===================')
        for item in tqdm(list_item):
            user_temp_list = set()  # check if user id happened in the item before
            new_idx_list = set()
            user_user_dict = dict()
            edge_set = set()
            user_map = dict()

            cascade_file = '{}.txt'.format(item)
            cascade_path = './{}/{}/{}/{}'.format(self.raw_dir, self.dataset, self.tree, cascade_file)
            cascade = pd.read_csv(cascade_path, sep='\n', header=None)

            for j in range(len(cascade)):
                str_line = cascade.iloc[j].values[0]
                former_uid, _, latter_uid, _ = self.extract_tree(str_line)

                if j == 0:
                    if former_uid != 'ROOT':
                        print("stop. at {}".format(item))
                    # assert former_uid == 'ROOT'
                    new_former_idx = global_user_id
                    new_idx_list.add(new_former_idx)
                    node_attribute.append(bert_dict.item()[item].flatten())
                    node_graph_id.append(graph_id)
                    global_user_id += 1

                    latter_idx = self.fetch_idx_with_uid(uid_idx_map, latter_uid)
                    assert latter_idx not in user_temp_list
                    user_temp_list.add(latter_idx)
                    # if latter_idx not in user_temp_list:
                    new_latter_idx = global_user_id
                    user_user_dict[latter_idx] = global_user_id
                    global_user_id += 1
                    new_idx_list.add(new_latter_idx)
                    latter_emb = self.fetch_emb_with_uid(user_emb, uid_idx_map, latter_uid)
                    node_attribute.append(latter_emb)
                    node_graph_id.append(graph_id)
                    A.append([new_former_idx, new_latter_idx])

                if j != 0:
                    former_idx = self.fetch_idx_with_uid(uid_idx_map, former_uid)
                    latter_idx = self.fetch_idx_with_uid(uid_idx_map, latter_uid)

                    if former_idx not in user_temp_list:
                        user_user_dict[former_idx] = global_user_id
                        global_user_id += 1
                        user_temp_list.add(former_idx)
                        new_idx_list.add(user_user_dict[former_idx])
                        former_emb = self.fetch_emb_with_uid(user_emb, uid_idx_map, former_uid)
                        node_attribute.append(former_emb)
                        user_map.update({user_user_dict[former_idx]: former_emb})
                        node_graph_id.append(graph_id)

                    if latter_idx not in user_temp_list:
                        user_user_dict[latter_idx] = global_user_id
                        global_user_id += 1
                        user_temp_list.add(latter_idx)
                        new_idx_list.add(user_user_dict[latter_idx])
                        latter_emb = self.fetch_emb_with_uid(user_emb, uid_idx_map, latter_uid)
                        node_attribute.append(latter_emb)
                        user_map.update({user_user_dict[latter_idx]: latter_emb})
                        node_graph_id.append(graph_id)

                    edge_set.add((user_user_dict[former_idx], user_user_dict[latter_idx]))
                    A.append([user_user_dict[former_idx], user_user_dict[latter_idx]])
            total_data.update(
                {
                    graph_id: {'user_list': new_idx_list,
                               'edge_set': edge_set,
                               'user_map': user_map,
                               'label': label_dict[item]
                               }
                }
            )
            graph_id += 1

        for g_id in range(graph_id):
            label.append(total_data[g_id]['label'])

        graph_id_list = [i for i in range(graph_id)]
        np.random.shuffle(graph_id_list)
        """train-val-test : 20-10-70"""
        train_idx = graph_id_list[:int(0.2 * len(graph_id_list))]
        val_idx = graph_id_list[int(0.2 * len(graph_id_list)): int(0.3 * len(graph_id_list))]
        test_idx = graph_id_list[int(0.3 * len(graph_id_list)):]
        """
        A: [[0,1],[0,2]...]
        node_graph_id: [0,0,0,1,1,1,2,2,3,3,3...]
        label: [0,3,1,1,3,0....]  (four types 0-3)
        """
        re_dir = './{}/re_{}'.format(self.raw_dir, self.dataset)
        if not os.path.exists(re_dir):
            os.mkdir(re_dir)
        # 1) A.txt
        A = np.array(A)

        sort_A = A[np.lexsort(A[:,::-1].T)]
        file = open('{}/A.txt'.format(re_dir), mode='w')
        for line in sort_A:
            file.write('{}, {}\n'.format(line[0], line[1]))
        file.close()

        # 2) graph_labels.npy
        path_graph_labels = '{}/graph_labels.npy'.format(re_dir)
        np.save(path_graph_labels, label)

        # 3) node_graph_id.npy
        path_node_graph_id = '{}/node_graph_id.npy'.format(re_dir)
        np.save(path_node_graph_id, node_graph_id)

        # 4) new_bert_feature.npz
        import scipy.sparse as sp
        sp_node_attribute = sp.csr_matrix(node_attribute)
        path_node_attribute = '{}/new_bert_feature.npz'.format(re_dir)
        sp.save_npz(path_node_attribute, sp_node_attribute)

        # 5) train-val-test_idx.npy
        path_train_idx = '{}/train_idx.npy'.format(re_dir)
        path_val_idx = '{}/val_idx.npy'.format(re_dir)
        path_test_idx = '{}/test_idx.npy'.format(re_dir)
        np.save(path_train_idx, train_idx)
        np.save(path_val_idx, val_idx)
        np.save(path_test_idx, test_idx)

        print('===================raw data has been generated=================')

    def generate_raw_data2(self):
        list_item = self.list_item()
        list_user = self.list_user()
        user_emb_path = './{}/{}/{}'.format(self.raw_dir, self.dataset, self.user_emb_file)
        user_emb = np.load(user_emb_path, allow_pickle=True)

        bert_user_emb_path = './{}/{}/{}'.format(self.raw_dir, self.dataset, self.bert_user_emb_file)
        bert_user_emb = np.load(bert_user_emb_path, allow_pickle=True)

        user_map_path = './{}/{}/{}'.format(self.raw_dir, self.dataset, self.data_map_file)
        uid_idx_map = np.load(user_map_path, allow_pickle=True)


        graph_id = 0
        graph_label = []

        label_dict = self._dict_label()
        bert_dict = self.fetch_bert()

        total_data = dict()
        # TODO: check type of this elem

        label = list()
        
        fold_NAME = 'graph'
        if not os.path.exists('./{}/{}/{}'.format(self.raw_dir, self.dataset, fold_NAME)):
           os.mkdir('./{}/{}/{}'.format(self.raw_dir, self.dataset, fold_NAME))
        print('======================generating raw data...===================')
        for item in tqdm(list_item):
            A = list()
            global_user_id = 0
            node_attribute = list()
            bert_node_attribute = list()

            user_temp_list = set()  # check if user id happened in the item before
            new_idx_list = set()
            user_user_dict = dict()
            edge_set = set()
            user_map = dict()
            node_graph_id = list()
            uid2localid = dict()

            cascade_file = '{}.txt'.format(item)
            cascade_path = './{}/{}/{}/{}'.format(self.raw_dir, self.dataset, self.tree, cascade_file)
            cascade = pd.read_csv(cascade_path, sep='\n', header=None)

            for j in range(len(cascade)):
                str_line = cascade.iloc[j].values[0]
                former_uid, _, latter_uid, _ = self.extract_tree(str_line)

                if j == 0:
                    if former_uid != 'ROOT':
                        print("stop. at {}".format(item))
                    uid2localid[former_uid] = global_user_id
                    node_graph_id.append(global_user_id)
                    global_user_id += 1

                    node_attribute.append(bert_dict.item()[item].flatten())
                    bert_node_attribute.append(bert_dict.item()[item].flatten())

                    if latter_uid not in uid2localid.keys():
                        uid2localid[latter_uid] = global_user_id
                        node_graph_id.append(global_user_id)
                        global_user_id += 1
                        node_attribute.append(self.fetch_emb_with_uid(user_emb, uid_idx_map, latter_uid))
                        bert_node_attribute.append(self.fetch_emb_with_uid(bert_user_emb, uid_idx_map, latter_uid))
                    A.append([0, 1])

                if j != 0:
                    if former_uid not in uid2localid.keys():
                        uid2localid[former_uid] = global_user_id
                        node_graph_id.append(global_user_id)
                        global_user_id += 1
                        node_attribute.append(self.fetch_emb_with_uid(user_emb, uid_idx_map, former_uid))
                        bert_node_attribute.append(self.fetch_emb_with_uid(bert_user_emb, uid_idx_map, former_uid))
                    if latter_uid not in uid2localid.keys():
                        uid2localid[latter_uid] = global_user_id
                        node_graph_id.append(global_user_id)
                        global_user_id += 1
                        node_attribute.append(self.fetch_emb_with_uid(user_emb, uid_idx_map, latter_uid))
                        bert_node_attribute.append(self.fetch_emb_with_uid(bert_user_emb, uid_idx_map, latter_uid))
                    A.append([uid2localid[former_uid], uid2localid[latter_uid]])


            saved_path = './{}/{}/{}/{}.npz'.format(self.raw_dir, self.dataset,fold_NAME, item)
            assert np.array(node_attribute).shape[1] == 768
            np.savez(saved_path,
                     x=np.array(node_attribute),
                     bert_x=np.array(bert_node_attribute),
                     edgeindex=A,
                     y=label_dict[item],
                     bert=bert_dict.item()[item].flatten()
                     # node_graph_id=np.array(node_graph_id)
                     )

        print('===================raw data has been generated=================')

class MLP(nn.Module):
    def __init__(self, n_layers, hid=10):
        super().__init__()
        self.n_layers = n_layers
        self.hid = hid
        self.ls = nn.ModuleList()
        for i in range(n_layers):
            if i:
                layer = nn.Linear(self.hid, self.hid)
            else:
                layer = nn.Linear(1, self.hid)
            self.ls.append(layer)    
        self.outlayer = nn.Linear(self.hid, 1)
        self.act = nn.LeakyReLU()
    def forward(self, x):
        x_tmp = copy.deepcopy(x)
        for i in range(self.n_layers):
            x_tmp = self.act(self.ls[i](x_tmp))
        x_tmp = self.outlayer(x_tmp)
        return x_tmp
        

class BPR(nn.Module):
    def __init__(self, user_size, item_size, dim, weight_decay, causal=False,  bert_item=None, user_emb_form=None, args=None):
        super().__init__()
        self.user_emb_form = user_emb_form
        self.raw_dir = args.raw_dir
        self.dataset = args.dataset
        self.bert_file = 'rumor_bert.npy'
        self.args = args
        assert self.user_emb_form in ['user_matrix', 'reweight_bert_sum','both']
        self.W = nn.Parameter(torch.empty(user_size, dim))
        nn.init.xavier_normal_(self.W.data)
        if bert_item is None:
            self.H = nn.Parameter(torch.empty(item_size, dim))
            nn.init.xavier_normal_(self.H.data)
        else:
            self.H = nn.Parameter(torch.tensor(bert_item))
            assert dim == 768

        # TODO: initialize item emb with bert feature
        
        self.weight_decay = weight_decay
        self.causal = causal
        self.mlp = MLP(args.n_layers, hid=args.h_mlp)
        self.data_item_map_file = args.data_item_map_file

        print('===========================BPR model parameters======================\n'
              'num_user: {}\n'
              'num_item: {}\n'
              'hidden_dim:{}\n'
              'causal:{}\n'.format(user_size, item_size, dim, causal))

    def forward(self, u, i, i_pop, j, j_pop):
        """Return loss value.
            :param u:
            :param i:
            :param i_pop:
            :param j:
            :param j_pop:
        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]
            i(torch.LongTensor): tensor stored item indexes which is prefered by user. [batch_size,]
            j(torch.LongTensor): tensor stored item indexes which is not prefered by user. [batch_size,]

        Returns:
            torch.FloatTensor

        """
        u = self.W[u, :]
        i = self.H[i, :]
        j = self.H[j, :]
        x_ui = torch.mul(u, i).sum(dim=1)
        x_uj = torch.mul(u, j).sum(dim=1)
        if self.causal:
            i_pop_pred = self.mlp(i_pop.float().view(-1,1).cuda()).flatten().cuda()
            j_pop_pred = self.mlp(j_pop.float().view(-1,1).cuda()).flatten().cuda()
            
            x_ui = x_ui * i_pop_pred.cuda()
            x_uj = x_uj * j_pop_pred.cuda()

        x_uij = x_ui - x_uj
        log_prob = F.logsigmoid(x_uij).sum()
        regularization = self.weight_decay * (
                u.norm(dim=1).pow(2).sum() + i.norm(dim=1).pow(2).sum() + j.norm(dim=1).pow(2).sum())
        return -log_prob + regularization

    def save_mlp(self):
        save_path = './saved_model/{}_n_layers_{}_h_mlp_{}.pth'.format(self.dataset, self.args.n_layers, self.args.h_mlp)
        state = {'args':self.args, 'state_dict': self.mlp.state_dict()}
        torch.save(state, save_path)

    def recommend(self, u):
        """Return recommended item list given users.

        Args:
            u(torch.LongTensor): tensor stored user indexes. [batch_size,]

        Returns:
            pred(torch.LongTensor): recommended item list sorted by preference. [batch_size, item_size]
        """
        u = self.W[u, :]
        x_ui = torch.mm(u, self.H.t())
        pred = torch.argsort(x_ui, dim=1)
        return pred

    def fetch_user_emb(self):
        assert self.user_emb_form in ['user_matrix', 'reweight_bert_sum', 'both']
        if self.user_emb_form == 'user_matrix':
            return self.W

        elif self.user_emb_form == 'reweight_bert_sum':
            rec_mat = torch.mm(self.W, self.H.t())
            content_bert = np.load('./{}/{}/{}'.format(self.raw_dir, self.dataset, self.bert_file), allow_pickle=True).item()
            item_map = np.load('./{}/{}/{}'.format(self.raw_dir, self.dataset, self.data_item_map_file), allow_pickle=True).item()

            bert_mat = []
            for idx in list(item_map.keys()):
                content_bert_i = content_bert[idx]
                bert_mat.append(content_bert_i)
            bert_mat = np.array(bert_mat)
            user_emb = []
            print('==================================reweight user emb with bert============================')
            for i in tqdm(range(rec_mat.shape[0])):
                tmp_list = []
                for j in range(rec_mat.shape[1]):
                    tmp_list.append(rec_mat[i][j]*bert_mat[j])
                user_emb_i = np.array(tmp_list).sum(axis=0) # shape should be (768,)
                user_emb.append(user_emb_i)
            return torch.from_numpy(np.array(user_emb))

        elif self.user_emb_form == 'both':
            rec_mat = torch.mm(self.W, self.H.t())
            content_bert = np.load('./{}/{}/{}'.format(self.raw_dir, self.dataset, self.bert_file),
                                   allow_pickle=True).item()
            item_map = np.load('./{}/{}/{}'.format(self.raw_dir, self.dataset, self.data_item_map_file),
                               allow_pickle=True).item()

            bert_mat = []
            for idx in list(item_map.keys()):
                content_bert_i = content_bert[idx]
                bert_mat.append(content_bert_i)
            bert_mat = np.array(bert_mat)
            user_emb = []
            print('=============================reweight user emb with bert=========================')

            for i in tqdm(range(rec_mat.shape[0])):
                tmp_list = []
                for j in range(rec_mat.shape[1]):
                    tmp_list.append(rec_mat[i][j].cpu().detach().numpy() * bert_mat[j])
                user_emb_i = np.array(tmp_list).sum(axis=0)  # shape should be (768,)
                user_emb.append(user_emb_i)
            return self.W, torch.from_numpy(np.array(user_emb))


def print_model_parm_nums(model, str):
    total_num = sum([param.nelement() for param in model.parameters()])
    print('{} params: {}'.format(str, total_num))

def main(args):
    uem = User_emb_model(args)

    list_data, user_mapping, item_mapping = uem.form_dict_data()
    dataset = TripletUniformPair(list_data, True, args.n_epochs)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16)
    user_size = len(uem.list_user())
    item_size = len(uem.list_item())
    bert_item = np.load('./{}/{}/rumor_bert.npy'.format(args.raw_dir, args.dataset), allow_pickle=True).item()
    bert_list = []
    for idx in bert_item.keys():
        bert_list.append(bert_item[idx])
    bert_list = np.array(bert_list).squeeze(axis=1)
    if not args.bert_init:
        bert_list = None
        print('Not use bert to init item matrix, dim is {}'.format(args.dim))
    else:
        assert args.dim == 768
        print('Use bert to init item matrix, dim is {}'.format(args.dim))
    model = BPR(user_size, item_size, args.dim, args.weight_decay, causal=args.causal,  bert_item=bert_list, user_emb_form=args.user_emb_form, args=args).cuda()
    print_model_parm_nums(model, BPR)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter('runs/n_layers_{}_h_mlp_{}'.format(args.n_layers,args.h_mlp))

    # Training
    smooth_loss = 0
    idx = 0
    for u, i, i_pop, j, j_pop in tqdm(loader, total=int(args.n_epochs * (len(list_data) / args.batch_size))):
        optimizer.zero_grad()
        loss = model(u, i, i_pop, j, j_pop)
        loss.backward()
        optimizer.step()
        writer.add_scalar('train/loss', loss, idx)
        smooth_loss = smooth_loss * 0.99 + loss * 0.01
        if idx % args.print_every == (args.print_every - 1):
            print('loss: %.4f\n' % smooth_loss)

        idx += 1
    print(loss, smooth_loss)
    print('Saving MLP...')
    model.save_mlp()
    
    if args.user_emb_form == 'both':
        user_emb, bert_user_emb = model.fetch_user_emb()
        user_emb = user_emb.cpu().detach().numpy()
        bert_user_emb = bert_user_emb.cpu().detach().numpy()
        save_path = './{}/{}/{}'.format(args.raw_dir, args.dataset,args.user_emb_file)
        bert_save_path = './{}/{}/{}'.format(args.raw_dir, args.dataset, args.bert_user_emb_file)
        np.save(save_path, user_emb)
        np.save(bert_save_path, bert_user_emb)
        print('saving both user emb and bert user emb\n dataset:{}'.format(args.dataset))
    if args.user_emb_form == 'user_matrix':
        user_emb = model.fetch_user_emb().cpu().detach().numpy()
        save_path = './{}/{}/{}'.format(args.raw_dir, args.dataset, args.user_emb_file)
        np.save(save_path, user_emb)
        print('saving user emb\n dataset:{}'.format(args.dataset))
    else:
        print('args.user_emb_form should be user_matrix or both')
    

if __name__ == '__main__':
    main(args=arg) # generate user embedding
    uem = User_emb_model(arg)
    uem.generate_raw_data2()
