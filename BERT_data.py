import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
# from dataloading import POSDataset, padding_collate_fn, IDX2POS, IGNORE_IDX


def encode_data(target, tokenizer):
    ids = []

    for sentences in target:
        encode = tokenizer.encode_plus(
            text=sentences,
            add_special_tokens=True,
            padding=True,
            max_length=103,
            pad_to_multiple_of=103,
            return_attention_mask=True,
            return_tensors='np',
            truncation=True
        )

        ids.append(encode['input_ids'])

    return ids

def create_token2idx(train_data):
    unique_tokens = []
    token2idx = {}
    index_cnt = 0

    # create a dict that does token2idx
    for tweet in train_data:
        for token in tweet:
            if token not in unique_tokens:
                unique_tokens.append(token)
                token2idx[token] = index_cnt
                index_cnt += 1

    return token2idx, index_cnt

def change_tokens2idx(data, token2idx, unknown_idx):
    for idx_tweet in range(len(data)):
        for idx_token in range(103):
            # check for tokens that are in test set but not in training
            if data[idx_tweet][idx_token] in token2idx.keys():
                data[idx_tweet][idx_token] = token2idx[data[idx_tweet][idx_token]]
            else:
                data[idx_tweet][idx_token] = unknown_idx
    return data


def read_tokenized_data():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    train_data = pd.read_csv('data/training.tsv', sep='\t')
    train_tweets = train_data['tweet'].to_numpy()
    train_labels = train_data['subtask_a'].to_numpy()
    test_tweets = pd.read_csv('data/testset-levela.tsv', sep='\t')
    test_labels = pd.read_csv('data/labels-levela.csv', header=None).to_numpy()[:, 1]

    train_ids = encode_data(train_tweets, tokenizer)
    test_ids = encode_data(test_tweets['tweet'], tokenizer)
    # get rid of the weird middle dim
    test_better_dim = []
    for i in range(len(test_ids)):
        test_better_dim.append(test_ids[i][0])
    dev_set = []
    for x in range(1000):
        dev_set.append(train_ids[x][0])
    train_better_dim = []
    for i in range(1000, len(train_ids)):
        train_better_dim.append(train_ids[i][0])
    # convert all tokens to indexes
    token2idx, unknown_idx = create_token2idx(train_better_dim)
    train_tweets = change_tokens2idx(train_better_dim, token2idx, unknown_idx)
    dev_tweets = change_tokens2idx(dev_set, token2idx, unknown_idx)
    test_tweets = change_tokens2idx(test_better_dim, token2idx, unknown_idx)

    dev_labels = train_labels[0:1000]
    train_labels = train_labels[1000:]

    training_set = [(train_tweets[index], train_labels[index]) for index in range(0, len(train_tweets) - 1)]
    dev_set1 = [(dev_tweets[index], dev_labels[index]) for index in range(0, len(dev_tweets) - 1)]
    test_set = [(test_tweets[index], test_labels[index]) for index in range(0, len(test_tweets) - 1)]

    return training_set, dev_set1, test_set

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    train_data = pd.read_csv('data/training.tsv', sep='\t')
    train_labels = train_data['subtask_a']
    test_tweets = pd.read_csv('data/testset-levela.tsv', sep='\t')
    test_labels = pd.read_csv('data/labels-levela.csv', header=None).to_numpy()[:, 1]

    train_ids = encode_data(train_data, tokenizer)
    test_ids = encode_data(test_tweets, tokenizer)

    dev_ids = train_ids[:1000]
    dev_labels = train_labels[:1000]

    train_ids = train_ids[1000:]
    train_labels = train_labels[1000:]

    df = pd.DataFrame({
        "train_encoding": train_ids,
        "train_labels": train_labels
    })
    df.to_csv('bert_train.csv')

    df = pd.DataFrame({
        "dev_encoding": dev_ids,
        "dev_labels": dev_labels
    })
    df.to_csv('bert_dev.csv')

    df = pd.DataFrame({
        "test_encoding": test_ids,
        "test_labels": test_labels
    })

    df.to_csv('bert_test.csv')