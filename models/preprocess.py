from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pandas as pd
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import operator 

def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
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
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def apply_preprocessing(dataset):
    table = str.maketrans('', '', "!#$%&'()*+-./’:;<=>?[\]^_`{|}~")
    dataset = [w.translate(table) for w in dataset]
    dataset = [word.lower() for word in dataset]
    dataset = [word_tokenize(sentence) for sentence in dataset]
    return dataset

dataset = pd.read_csv('data/training.tsv', sep='\t')
all_tweets = dataset['tweet'].to_numpy()
all_labels = dataset['subtask_a'].to_numpy()
test_tweets = pd.read_csv('data/testset-levela.tsv', sep='\t').to_numpy()[:,1]
test_labels = pd.read_csv('data/labels-levela.csv').to_numpy()[:,1]

sentences = [word_tokenize(sentence) for sentence in all_tweets]
vocab = build_vocab(sentences)
print({k: vocab[k] for k in list(vocab)[:50]})
#print(sentences)

#import GLOVE cleaned up glove embeddings
glovepath = 'embeds.txt'
print('loading embeddings...')
embeddings_index = KeyedVectors.load_word2vec_format(glovepath, binary=False)
oov = check_coverage(vocab,embeddings_index)

#remove punctuations,lowerize words, tokenize sentences into words
all_tweets = apply_preprocessing(all_tweets)
test_tweets = apply_preprocessing(test_tweets)
print(test_tweets[0])

#check embeddings for words in dataset after preprocessing
newvocab=build_vocab(all_tweets)
check_coverage(newvocab,embeddings_index)

print(newvocab)
#data split into dev and train
dev_tweets = all_tweets[:1000]
dev_labels = all_labels[:1000]
train_tweets = all_tweets[1000:]
train_labels = all_labels[1000:]
