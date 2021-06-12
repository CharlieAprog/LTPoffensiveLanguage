from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import sent_tokenize
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import operator 
import torch
from torch.utils.data import DataLoader, Dataset 
import numpy as np
IGNORE_IDX = -100

def check_coverage(vocab,embeddings_index):
    print('checking coverage...')
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in vocab:
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]
    return sorted_x

def build_vocab(sentences, verbose =  True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    print('building vocab...')
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def apply_preprocessing(dataset):
    print('preprocessing...')
    table = str.maketrans('', '', "!#$%&'()*+-./’:;<=>?[\]^_`{|}~")
    #mapping = str.maketrans('twitteruser', "@USER")
    dataset = [w.replace("@USER","twitteruser") for w in dataset]
    dataset = [w.translate(table) for w in dataset]
    dataset = [word.lower() for word in dataset]
    dataset = [word_tokenize(sentence) for sentence in dataset]
    return dataset

def gather_words(all_tweets, test_tweets):
    print('gathering words...')
    words = []
    for tweet in apply_preprocessing(all_tweets):
        for word in tweet:
            words.append(word)
        
    for tweet in apply_preprocessing(test_tweets):
        for word in tweet:
            words.append(word)
    return words

def clean_embeddings(path, words):
    embedded_words = []
    embedded_vecs = []
        
    text_file = open(path, "r")
    lines = text_file.readlines()
    for line in tqdm(lines):
        line = line.split(' ')
        word = line[0]
        if word in words and word not in embedded_words:
            embedded_words.append(word)
            embedded_vecs.append(line[1:])

    size = len(embedded_words)
    f = open("embeds.txt", "w+")
    f.write(f'{size} 25\n')
    for i in tqdm(range(size)):
        f.write(f'{embedded_words[i]} ') 
        for element in embedded_vecs[i]:
            f.write(f'{element} ') 
        f.write(f'\n')
    f.close()
    
def get_embedded_words(path):
    print('get embedded words...')
    text_file = open(path, "r",encoding = 'utf-8' )
    lines = text_file.readlines()
    embedded_words = []
    embedded_vecs = []
    for line in lines:
        line = line.split(' ')
        embedded_words.append(line[0])
        embedded_vecs.append(line[1:])
    return embedded_words, embedded_vecs
        
def print_missing_embeds(words, embedded_words):
    for word in words:
        if word not in embedded_words:
            print(word)
    pass

def ignore_words(tweets, embedded_words):
    print('ignoring unknown words...')
    for i in tqdm(range(len(tweets))):
        for j in range(len(tweets[i])):
            if tweets[i][j] not in embedded_words:
                tweets[i][j] = 'ignoreword'


def load_data():

    print('reading data...')
    dataset = pd.read_csv('data/training.tsv', sep='\t')
    all_tweets = dataset['tweet'].to_numpy()
    all_labels = dataset['subtask_a'].to_numpy()
    test_tweets = pd.read_csv('data/testset-levela.tsv', sep='\t').to_numpy()[:,1]
    test_labels = pd.read_csv('data/labels-levela.csv').to_numpy()[:,1]

    #load embeddings
    glovepath = 'embeds.txt'
    print('loading embeddings...')
    embeddings = KeyedVectors.load_word2vec_format(glovepath, binary=False)
    embedded_words, embedded_vectors = get_embedded_words(glovepath)

    #preprocess tweets i.e lowercase, remove punctuation, remove words that are not in embeddings
    all_tweets = apply_preprocessing(all_tweets)
    max_len = 0
    for tweet in all_tweets:
        lenx = len(tweet)
        if lenx > max_len:
            max_len = lenx
    print(max_len)
    test_tweets = apply_preprocessing(test_tweets)
    ignore_words(all_tweets, embedded_words)
    ignore_words(test_tweets, embedded_words)

    #vocabulary of dataset
    newvocab=build_vocab(all_tweets)
    check_coverage(newvocab,embeddings)
    dev_tweets = all_tweets[:1000]
    dev_labels = all_labels[:1000]
    train_tweets = all_tweets[1000:]
    train_labels = all_labels[1000:]
    training_set = [(train_tweets[index],train_labels[index]) for index in range(len(train_tweets)-1)]
    dev_set =  [(dev_tweets[index],dev_labels[index]) for index in range(len(dev_labels)-1)]
    test_set =  [(test_tweets[index],test_labels[index]) for index in range(len(test_labels)-1)]
    return training_set, dev_set, test_set, newvocab, embeddings

class MyList(list):
    def ljust(self, n, fillvalue=''):
        return self + [fillvalue] * (n - len(self))

def padding_collate_fn(batch):
    """ Pads data with zeros to size of longest sentence in batch. """
    data, labels = zip(*batch)
    largest_sample = max([len(d) for d in data])
    padded_data = []
    for i, sample in enumerate(data):
        new_list = MyList(sample)
        padded_data.append(new_list.ljust(largest_sample, 'ignoreword'))
    return padded_data, labels

