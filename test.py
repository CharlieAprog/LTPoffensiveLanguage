import pandas as pd
from tqdm import tqdm
from models.preprocess import apply_preprocessing


# embed  = pd.read_csv('data/word2vec/25d.txt', sep=' ')
# print(embed.head)

def gather_words(all_tweets, test_tweets):
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
    text_file = open(path, "r")
    lines = text_file.readlines()
    embedded_words = []
    embedded_vecs = []
    for line in tqdm(lines):
        line = line.split(' ')
        embedded_words.append(line[0])
        embedded_vecs.append(line[1:])
    return embedded_words, embedded_vecs
        

def print_missing_embeds(words, embedded_words):
    for word in words:
        if word not in embedded_words:
            print(word)
    pass

dataset = pd.read_csv('data/training.tsv', sep='\t')
test_tweets = pd.read_csv('data/testset-levela.tsv', sep='\t').to_numpy()[:,1]
all_tweets = dataset['tweet'].to_numpy()
all_labels = dataset['subtask_a'].to_numpy()
training_tweets = all_tweets[1000:]

path = 'data/word2vec/25d.txt'
embedded_path = 'embeds.txt'
words = gather_words(all_tweets, test_tweets)
# clean_embeddings(path, words)
embedded_words, embedded_vecs = get_embedded_words(embedded_path)
print_missing_embeds(words, embedded_words)

