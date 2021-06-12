from numpy.lib import twodim_base
import pandas as pd
import numpy as np
import torch
from torch import nn
from preprocess import * 
from gensim.models import KeyedVectors
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
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers, batch_first = True) #(for batch fist we need to give the model input size (batchsize, sequence length, input_dim)
        self.fc =  nn.Linear(hidden_size,num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        out, _ = self.lstm(x, (h0,c0)) 
        
        out = out[:,-1, :]
        out = self.fc(out)
        
        return out

def convert_into_embeddings(data):
    embdata= []
    for sentences in data:
        sentence = []
        for words in sentences:
            sentence.append(embeddings[words])
        embdata.append(sentence)
    return embdata

def train(model,data_loader,device, criterion,optimizer,embeddings, num_epochs):
    # Train the model
    n_total_steps = len(data_loader)
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_loader):
            
            embdata = convert_into_embeddings(data)
            #data = [embeddings.get_vector(word) for word in sentence for sentence in data]

            x_tensor = torch.tensor(embdata, dtype=torch.float32).to(device)
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
        test(model,dev_loader,device,criterion,optimizer,embeddings)

def test(model,dev_loader,device,criterion,optimizer,embeddings):
    # Test the model \w (dev_loader)
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for data, labels in dev_loader:
            data = convert_into_embeddings(data).to(device)
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
    #load
    training_set, dev_set, test_set, vocab, embeddings = load_data()

    #Data Loaders
    batch_size = 2
    input_size = 25
    learning_rate = 0.0002
    hidden_size = 200
    num_layers = 2
    num_classes = 2
    num_epochs = 1

    train_loader = torch.utils.data.DataLoader(dataset=training_set, collate_fn=padding_collate_fn, batch_size=batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dataset=dev_set, collate_fn=padding_collate_fn, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set,collate_fn=padding_collate_fn, batch_size=batch_size, shuffle=True)

    model = LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

    #Hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
    torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    train(model,train_loader,device,criterion,optimizer,embeddings, num_epochs)


    # Test the model \w (dev_loader)
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


