import pandas as pd
import numpy as np
dataset = pd.read_csv('data/training.tsv', sep='\t')
training_tweeets = dataset[['tweet']].to_numpy()
training_labels = dataset[['subtask_a']].to_numpy()
#TODO set up validation set
# set up testing set
# set up 