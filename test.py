import pandas as pd
from tqdm import tqdm
dataset = pd.read_csv('data/training.tsv', sep='\t')
test_tweets = pd.read_csv('data/testset-levela.tsv', sep='\t').to_numpy()[:,1]
all_tweets = dataset['tweet'].to_numpy()
all_labels = dataset['subtask_a'].to_numpy()
training_tweets = all_tweets[1000:]

# embed  = pd.read_csv('data/word2vec/25d.txt', sep=' ')
# print(embed.head)

words = []
test_words= []
for tweet in all_tweets:
    for word in tweet.split(' '):
        words.append(word)
    
for tweet in test_tweets:
    for word in tweet.split(' '):
        words.append(word)

embedded_words = []
embedded_vecs = []

text_file = open('data/word2vec/25d.txt', "r")
lines = text_file.readlines()
count = 0
for line in lines:
    if count % 1000 == 0:
        print(count/119354)
    count += 1
    line = line.split(' ')
    if (line[0] in words or line[0] in test_words) and line[0] not in embedded_words:
        embedded_words.append(line[0])
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