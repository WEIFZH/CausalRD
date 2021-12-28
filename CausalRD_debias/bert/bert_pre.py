"""
[ONE.]
output:         rumor_bert.npy

discription:    use bert-as-service(https://github.com/hanxiao/bert-as-service)

"""

import argparse
from bert_serving.client import BertClient
import pandas as pd
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--raw_dir', type=str, default='dataset')
parser.add_argument('--dataset', type=str, default='twitter16', choices=['twitter15', 'twitter16'])
parser.add_argument('--raw_file', type=str, default='source_tweets.txt')
parser.add_argument('--saved_file', type=str, default='rumor_bert.npy')
arg = parser.parse_args()


class bert_emb:
    def __init__(self, args):
        self.raw_dir = args.raw_dir
        self.dataset = args.dataset
        self.raw_file = args.raw_file
        self.saved_file = args.saved_file

    def process(self):
        dic_bert_content = {}
        bc = BertClient()
        file_path = '../{}/{}/{}'.format(self.raw_dir, self.dataset, self.raw_file)
        print('Processing file, in {}...'.format(file_path))
        f = open(file_path, "r")
        lines = f.readlines()
        for str_content in lines:
            print(str_content)
            tid, content = str_content.split('\t')
            content = content.split('\n')[0]
            res_content = bc.encode([content])
            assert dic_bert_content.get(tid) is None
            dic_bert_content[tid] = res_content
        f.close()
        save_path = '../{}/{}/{}'.format(self.raw_dir, self.dataset, self.saved_file)
        np.save(save_path, dic_bert_content)
        print('Finished, file was saved at {}.'.format(save_path))


if __name__ == '__main__':
    be = bert_emb(arg)
    be.process()
    """
    tid: bert_emb_content
    """

