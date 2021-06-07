# Imports
import torch
import torchvision  # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For a nice progress bar!
from preprocess import * 
from gensim.models import KeyedVectors
from numpy.lib import twodim_base
import pandas as pd


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #load
    training_set, dev_set, test_set, vocab, embeddings = load_data()
    train_tweets, train_labels = training_set
    dev_tweets, dev_labels = dev_set
    test_tweets, test_labels = test_set
    
    #try out some stuff bro
    print(len(train_tweets[0]))
    print(len(train_tweets[1]))
    print(len(train_tweets))
    print(embeddings['nigger'])
    #Data Loaders
    batch_size = 1
    train_loader = torch.utils.data.DataLoader(dataset=train_tweets, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    dev_loader = torch.utils.data.DataLoader(dataset=dev_tweets, 
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_tweets, 
                                            batch_size=batch_size, 
                                            shuffle=False)

    #examples = iter(train_loader)
    #example_data, example_targets = examples.next()

    # Hyperparameters
    input_size = 25
    hidden_size = 1
    num_layers = 2
    num_classes = 10
    sequence_length = 3
    learning_rate = 0.005
    batch_size = 64
    num_epochs = 3

    model = RNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
    for epoch in range(num_epochs):
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            # Get data to cuda if possible
            data = data.to(device=device).squeeze(1) #?? chief
            targets = targets.to(device=device)

            # forward
            scores = model(embeddings[data])
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent update step/adam step
            optimizer.step()

        print(f'Epoch : {epoch}')
        print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
        print(f"Accuracy on dev set: {check_accuracy(dev_loader, model)*100:.2f}")
    # Check accuracy on training & test to see how good our model
    def check_accuracy(loader, model):
        num_correct = 0
        num_samples = 0

        # Set model to eval
        model.eval()

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device=device).squeeze(1)
                y = y.to(device=device)

                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

        # Toggle model back to train
        model.train()
        return num_correct / num_samples

    print('Training is done!')
    print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
    print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")