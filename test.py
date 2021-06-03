import pandas as pd
data = pd.read_csv('data/training.tsv', sep='\t')
print(data['subtask_a'].value_counts())
print(data)

