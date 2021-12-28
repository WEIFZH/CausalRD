"""
[TWO.]
Generating 3 npy files for Twitter15 or Twitter16
1) postive_samples.npy
{
    user_id1: {
                item_id: popularity,
                item_id: popularity....
             },
             ...
}


2) negtive_samples.npy


3) user_former.npy

{
    user_id1: {
                user_id2, user_id3,...
            }//not sure list or dict, you can print it to check.
            ,...
}
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, choices=['twitter15', 'twitter16'], default='twitter15')
args = parser.parse_args()


class preprocess:
    def __init__(self, dataset='twitter15'):
        self.label2id = {'false': 0, 'true': 1, 'non-rumor': 2, 'unverified': 3}
        self.data_dir = 'dataset/{}'.format(dataset)
        self.label_file = 'label.txt'
        self.content_file = 'source_tweets.txt'
        self.tree_dir = 'tree'

        self.ut_name = 'positive_samples'
        self.uu_name = 'user_former'
        self.neg_name = 'negtive_samples'
        self.ut = {}  # users and their corresponding tweet, i.e., positive sample
        self.uu = {}  # users and their former user

        print('Processing {}...'.format(dataset))

    def _dict_label(self):
        """
        :return dictionary of labels
        """
        label_path = './{}/{}'.format(self.data_dir, self.label_file)
        labels = pd.read_csv(label_path, header=None)
        label_dict = {}
        for i in range(len(labels)):
            str_label = labels.iloc[i].values[0]
            label, t_id = str_label.split(':')
            label_int = self.label2id[label]
            label_dict[t_id] = label_int

        return label_dict

    def _list_tweetid(self):
        return list(self._dict_label().keys())

    def extract_tree(self, str_line):
        # ['972651', '80080680482123777', '0.0']->['189397006', '80080680482123777', '1.8']
        former, latter = str_line.split('->')

        former_uid, former_tid, _ = former.split(',')
        latter_uid, latter_tid, _ = latter.split(',')
        former_uid = former_uid[2:-1]
        former_tid = former_tid[2:-1]

        latter_uid = latter_uid[2:-1]
        latter_tid = latter_tid[2:-1]

        return former_uid, former_tid, latter_uid, latter_tid

    def update_ut(self, uid, tid, pop):
        if self.ut.get(uid) is None:
            self.ut[uid] = {tid: pop}
        else:
            if self.ut[uid].get(tid) is None:
                self.ut[uid][tid] = pop

    def update_uu(self, uid, former_uid):
        # if uid == '5402612':
        #     print("stop")
        if former_uid == uid or former_uid == 'ROOT':
            return
        if self.uu.get(uid) is None:
            self.uu[uid] = [former_uid]
        else:
            if former_uid not in self.uu[uid]:
                self.uu[uid].append(former_uid)

    def postive_sample_data(self):
        list_tweetid = self._list_tweetid()

        for i in tqdm(list_tweetid):
            tweetid = i
            cascade_file = '{}.txt'.format(tweetid)
            cascade_path = './{}/{}/{}'.format(self.data_dir, self.tree_dir, cascade_file)
            cascade = pd.read_csv(cascade_path, sep='\n', header=None)

            for j in range(len(cascade)):
                str_line = cascade.iloc[j].values[0]
                if j == 0:
                    former_uid, former_tid, latter_uid, latter_tid = self.extract_tree(str_line)
                    # pop的计算方式有待商榷
                    self.update_ut(latter_uid, tweetid, j+1)
                else:
                    former_uid, former_tid, latter_uid, latter_tid = self.extract_tree(str_line)
                    self.update_ut(latter_uid, tweetid, j+1)
                    self.update_uu(latter_uid, former_uid)

        return self.ut, self.uu

    def search_tw(self, uid):
        return list(self.ut[uid].keys())
        # return self.ut[uid]

    def neg_sample(self):
        if not os.path.exists('./{}/{}.npy'.format(self.data_dir, self.ut_name)):
            self.postive_sample_data()
        else:
            self.ut = np.load('./{}/{}.npy'.format(self.data_dir, self.ut_name), allow_pickle=True).item()
            self.uu = np.load('./{}/{}.npy'.format(self.data_dir, self.uu_name), allow_pickle=True).item()
        neg_matrix = {}

        assert len(self.uu.keys()) == len(set(self.uu.keys()))

        ind = 0
        for i in tqdm(self.uu.keys()):
            # print(i)
            # if len(self.uu[i]) == 1:
            #     continue
            list_of_tid = []
            for j in self.uu[i]:
                # when key corruption occurs, select minimal value

                list_of_tid.append(self.ut[j])
            new_dic = self.merge_d(list_of_tid)

            for ele in list(self.ut[i].keys()):
                if new_dic.get(ele):
                    new_dic.pop(ele)

            # TODO: new_dic could be None?
            neg_matrix[i] = new_dic

        neg_path = './{}/{}.npy'.format(self.data_dir, self.neg_name)
        np.save(neg_path, neg_matrix)
        print('negtive samples have been saved in {}.'.format(neg_path))
        return neg_matrix

    def save_positive_sample(self):
        # ref:    cnblogs.com/shanyr/p/11243889.html
        ut, uu = self.postive_sample_data()
        ut_save_path = './{}/{}.npy'.format(self.data_dir, self.ut_name)
        uu_save_path = './{}/{}.npy'.format(self.data_dir, self.uu_name)
        np.save(ut_save_path, ut)
        np.save(uu_save_path, uu)
        print("positive_samples and user_former have been saved in {} and {}".format(ut_save_path, uu_save_path))

    def merge_d(self, list_of_dic):
        new_dic = {}
        for dic in list_of_dic:
            for k, v in dic.items():
                if new_dic.get(k) is None:
                    new_dic[k] = v
                else:
                    new_dic[k] = min(v, new_dic[k])
        return new_dic


if __name__ == '__main__':
    pre = preprocess(dataset=args.dataset)
    pre.save_positive_sample()
    pre.neg_sample()

