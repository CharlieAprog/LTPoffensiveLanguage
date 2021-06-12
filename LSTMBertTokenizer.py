from numpy.lib import twodim_base
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from preprocess import * 
from BERT_data import *

def tensor_desc(x):
    """ Inspects a tensor: prints its type, shape and content"""
    print("Type:   {}".format(x.type()))
    print("Size:   {}".format(x.size()))
    print("Values: {}".format(x))

def get_label(y):
    if y == 'OFF':
        return 1
    return 0

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, num_layers, num_classes, dropout = 0.0):
        super(LSTM, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers, batch_first = True, dropout = dropout) #(for batch fist we need to give the model input size (batchsize, sequence length, input_dim)
        self.token_embedding = nn.Embedding(embedding_dim=input_size, num_embeddings=vocab_size)
        self.fc =  nn.Linear(hidden_size,num_classes)
    
    def forward(self, x):
        x = self.token_embedding(x)
        print(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        out, _ = self.lstm(x, (h0,c0)) 
        
        out = out[:,-1, :]
        out = self.fc(out)
        out = F.softmax(out)
        return out

def train(model,train_dl,dev_dl,device, criterion,optimizer, num_epochs):
    # Train the model
    n_total_steps = len(train_dl)
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_dl):
            x_tensor = torch.tensor(data).to(device)
            y_tensor = torch.tensor([get_label(label[0]) for label in labels]).to(device)
            # Forward pass; if IGNORE skip word, else feed model with embedding
            outputs = model(x_tensor) #(batchsize, sequencesize, vectorsize)
            loss = criterion(outputs, y_tensor)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        test(model,dev_dl,device,criterion,optimizer)

def test(model,dev_dl,device,criterion,optimizer):
    # Test the model \w (dev_loader)
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for data, labels in dev_dl:
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #Data Loaders
    batch_size = 2
    embed_size = 200
    hidden_size = 200
    num_layers = 3
    vocab_size = 14297
    learning_rate = 0.0002
    num_classes = 2
    num_epochs = 1
    dropout = 0.5

    #load data and create dataloaders
    train_data, dev_data, test_data = read_tokenized_data()
    train_dl = torch.utils.data.DataLoader(dataset=train_data, collate_fn=padding_collate_fn, batch_size=batch_size, shuffle=True)
    dev_dl = torch.utils.data.DataLoader(dataset=dev_data, collate_fn=padding_collate_fn, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(dataset=test_data, collate_fn=padding_collate_fn, batch_size=batch_size, shuffle=True)

    model = LSTM(embed_size, hidden_size, vocab_size, num_layers, num_classes, dropout = dropout).to(device)
    #example batch
    data,label  =  next(iter(train_dl))
    print(data)
    
    #Hyperparameters
    criterion = nn.CrossEntropyLoss(ignore_index = 26)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train(model,train_dl,dev_dl,device,criterion,optimizer, num_epochs)


    # Test the model \w (dev_loader)
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_dl:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')


