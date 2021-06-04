import pandas as pd
data = pd.read_csv('data/training.tsv', sep='\t')
print(data['subtask_a'].value_counts())
print(data)
print(data.columns)

embed  = pd.read_csv('data/word2vec/25d.txt', sep=' ')
print(embed.head)


text_file = open("wiki10k", "r")
lines = text_file.readlines()
for line in lines[1:1001]:
    line = line.split(' ')
    wiki_words.append(line[0])
    wiki_emb.append(line[1:])