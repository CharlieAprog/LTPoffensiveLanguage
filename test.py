import pandas as pd
from tqdm import tqdm
from models.preprocess import apply_preprocessing


# embed  = pd.read_csv('data/word2vec/25d.txt', sep=' ')
# print(embed.head)



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

