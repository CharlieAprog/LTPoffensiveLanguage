import pandas as pd
from tqdm import tqdm
from models.preprocess import apply_preprocessing
dataset = pd.read_csv('data/training.tsv', sep='\t')
test_tweets = pd.read_csv('data/testset-levela.tsv', sep='\t').to_numpy()[:,1]
all_tweets = dataset['tweet'].to_numpy()
all_labels = dataset['subtask_a'].to_numpy()
training_tweets = all_tweets[1000:]

# embed  = pd.read_csv('data/word2vec/25d.txt', sep=' ')
# print(embed.head)

words = []
test_words= []
for tweet in apply_preprocessing(all_tweets):
    for word in tweet:
        words.append(word)
    
for tweet in apply_preprocessing(test_tweets):
    for word in tweet:
        words.append(word)

embedded_words = []
embedded_vecs = []

text_file = open('embeds.txt', "r")
lines = text_file.readlines()
count = 0

for line in lines:
    line = line.split(' ')
    embedded_words.append(line[0])

for word in words:
    if word not in embedded_words:
        print(word)
    

text_file = open('data/word2vec/25d.txt', "r")
lines = text_file.readlines()
for line in tqdm(lines):
    line = line.split(' ')
    word = line[0]
    if word in words and word not in embedded_words:
        embedded_words.append(word)
        embedded_vecs.append(line[1:])

size = len(embedded_words)
print(size)

f = open("embeds.txt", "w+")
f.write(f'{size} 25\n')
for i in tqdm(range(size)):
    f.write(f'{embedded_words[i]} ') 
    for element in embedded_vecs[i]:
        f.write(f'{element} ') 
    f.write(f'\n')
f.close()