import pandas as pd
import numpy as np
import torch
from torch import nn

dataset = pd.read_csv('data/training.tsv', sep='\t')
all_tweets = dataset['tweet'].to_numpy()
all_labels = dataset['subtask_a'].to_numpy()

test_tweets = pd.read_csv('data/testset-levela.tsv', sep='\t').to_numpy()[:,1]
test_labels = pd.read_csv('data/labels-levela.csv').to_numpy()[:,1]
dev_tweets = all_tweets[:1000]
dev_labels = all_labels[:1000]
training_tweets = all_tweets[1000:]
training_labels = all_labels[1000:]

#lower case and tokenize
print(training_tweets.shape)
training_tweets = [word_tokenize(sent) for sent in training_tweets]
for sentences in training_tweets:
    for words in sentences:
        words = words.lower()
        print(words,'aaaaaaa')
print(training_tweets[0])


#TODO set up validation set
# set up testing set
# set up 
batch_size = 1

#Data loader
train_loader = torch.utils.data.DataLoader(dataset=training_tweets, 
                                           batch_size=batch_size, 
                                           shuffle=True)

dev_loader = torch.utils.data.DataLoader(dataset=dev_tweets, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_tweets, 
                                          batch_size=batch_size, 
                                          shuffle=False)

examples = iter(test_loader)
example_data, example_targets = examples.next()


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers, batch_first = True) #(for batch fist we need to give the model input size (batchsize, sequence length, input_dim)
        self.fc =  nn.Linear(hidden_size,num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        out, _ = self.lstm(x,h0)
        
        out = out[:,-1, :]
        out = self.fc(out)
        
        return out

#model = LSTM()

#load embeddings
embeddings = gensim.models.KeyedVectors.load_word2vec_format('data\word2vec\glove.twitter.27B.25d.txt')



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  


n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(train_loader):  
        #convert to embeddings
        data = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')