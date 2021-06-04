import pandas as pd
import numpy as np
dataset = pd.read_csv('data/training.tsv', sep='\t')
training_tweeets = self.dataset[['tweet']].to_numpy()
training_labels = self.dataset[['subtask_a']].to_numpy()
#TODO set up validation set
# set up testing set
# set up 