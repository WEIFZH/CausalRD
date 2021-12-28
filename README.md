# CausalRD
This repo provides a reference implementation of **CausalRD: A Causal View of Rumor Detection via Eliminating Popularity and Conformity Biases**

## Dependencies
python 3.7

pytorch 1.8.1

pytorch_geometric 1.7.0

bert-as-service 1.10.0

## Dataset
We use the [raw data](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0) in debias phase and pre-processed data released by [Bi-GCN](https://github.com/TianBian95/BiGCN) in inference phase. You can download the raw data and move it to ./CausalRD_debias/data/

## Usage

### Debias Phase

1. Preprocessing

We first leverage [bert-as-service](https://github.com/hanxiao/bert-as-service) to get the embeddings of source tweets. We use [BERT-Base](https://github.com/google-research/bert) (uncased_L-12_H-768_A-12) as the bert model. Please download it from above link and unzip it to bert/, create bert-as-service server and run the client.

```shell
pip install bert-serving-server  # server
pip install bert-serving-client  # client, independent of `bert-serving-server`
bert-serving-start -model_dir ./CasualRD/CausalRD_debias/bert/uncased_L-12_H-768_A-12/ -num_worker=4 
python ./CasualRD/CausalRD_debias/bert/bert_pre.py
```

2. Structral Negative Sampling

We then generate positive and negative samples by Structral Negative Sampling strategy.

```shell
python ./CausalRD/CausalRD_debias/sampling.py
```

3. Optimizing

```shell
python ./CausalRD/CausalRD_debias/optimizing.py --dataset twitter15 --n_layers 7
```
This will generate a directory ./graph/, move it to ./CausalRD/CausalRD_inference/data/Twitter15

### Inference Phase
create "Twitter15graph" folder and "Twitter16graph" folder in the ./CausalRD_inference/data folder

```shell
python ./CausalRD/CausalRD_inference/Process/getTwittergraph.py Twitter15
python ./CausalRD/CausalRD_inference/train.py --datasetname Twitter15 --weight_decay 0.0001 --iteration 100 --lr 0.0005
```

