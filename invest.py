import pandas as pd

data = pd.read_csv('olid.tsv', sep='\t')
print(data.head)
tweet_col = data.iloc[:, 1]
my_col = data.iloc[:, 2]
print(my_col.shape)
bad_cnt = 0
not_cnt = 0
off_cnt = 0
for idx in range(13240):
    print(tweet_col[idx])
    print(my_col[idx])
    if my_col[idx] == 'NULL':
        bad_cnt += 1
    elif my_col[idx] == 'NOT':
        not_cnt += 1
    elif my_col[idx] == 'OFF':
        off_cnt += 1

print(bad_cnt, not_cnt, off_cnt)
