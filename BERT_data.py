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
            return_tensors='pt',
            truncation=True
        )

        ids.append(encode['input_ids'])

    return ids


def read_tokenized_data():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    train_data = pd.read_csv('data/training.tsv', sep='\t')
    train_tweets = train_data['tweet']
    train_labels = train_data['subtask_a']
    test_tweets = pd.read_csv('data/testset-levela.tsv', sep='\t')
    test_labels = pd.read_csv('data/labels-levela.csv', header=None).to_numpy()[:, 1]

    train_ids = encode_data(train_tweets, tokenizer)
    test_ids = encode_data(test_tweets['tweet'], tokenizer)

    training_set = [(train_ids[index], train_labels[index]) for index in range(0, len(train_tweets) - 1)]
    test_set = [(test_ids[index], test_labels[index]) for index in range(0, len(test_tweets['tweet']) - 1)]

    return training_set, test_set

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    dataset = pd.read_csv('data/training.tsv', sep='\t')
    all_tweets = dataset['tweet'].to_numpy()
    all_labels = dataset['subtask_a'].to_numpy()
    test_tweets = pd.read_csv('data/testset-levela.tsv', sep='\t').to_numpy()[:, 1]
    test_labels = pd.read_csv('data/labels-levela.csv').to_numpy()[:, 1]

    train_ids = encode_data(train_data)
    test_ids = encode_data(test_tweets)

    dev_ids = train_ids[:1000]
    dev_labels = train_labels[:1000]

    train_ids = train_ids[1000:]
    train_labels = train_labels[1000:]
    print(train_ids)
    df = pd.DataFrame({
        "train_encoding": train_ids,
        "train_labels": train_labels
    })
    # df.to_csv('bert_train.csv')

    df = pd.DataFrame({
        "dev_encoding": dev_ids,
        "dev_labels": dev_labels
    })
    # df.to_csv('bert_dev.csv')

    df = pd.DataFrame({
        "test_encoding": test_ids,
        "test_labels": test_labels
    })

    # df.to_csv('bert_test.csv')