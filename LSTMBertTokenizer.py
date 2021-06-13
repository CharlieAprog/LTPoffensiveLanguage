from numpy.lib import twodim_base
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
import torch
from torch import nn
import torch.nn.functional as F
from preprocess import * 
from BERT_data import *
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix

def tensor_desc(x):
    """ Inspects a tensor: prints its type, shape and content"""
    print("Type:   {}".format(x.type()))
    print("Size:   {}".format(x.size()))
    print("Values: {}".format(x))

def get_label(y):
    if y == 'OFF':
        return 1
    return 0

def pad_collate(batch):
  (xx, yy) = zip(*batch)
  x_lens = [len(x) for x in xx]
  y_lens = [len(y) for y in yy]
  print(x_lens)
  print(xx,yy)
  xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=26)

  return xx_pad, yy_pad, x_lens, y_lens

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, vocab_size, num_layers, num_classes, dropout = 0.0):
        super(LSTM, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers, batch_first = True, dropout = dropout) #(for batch fist we need to give the model input size (batchsize, sequence length, input_dim)
        self.token_embedding = nn.Embedding(embedding_dim=input_size, num_embeddings=vocab_size)
        self.fc =  nn.Linear(hidden_size,num_classes)
    
    def forward(self, x):
        seqlen = [((x.size()[1] - (batch==26).sum())) for batch in x]
        x = self.token_embedding(x)
        xpacked = pack_padded_sequence(x, [lis.cpu() for lis in seqlen] , batch_first=True, enforce_sorted= False)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        wtf, (out,_) = self.lstm(xpacked, (h0,c0)) 
        
        out = self.fc(out[-1])
        out = torch.sigmoid(out)
        return out
    

def train(model,train_dl,dev_dl,device, criterion,optimizer, num_epochs):
    # Train the model
    n_total_steps = len(train_dl)
    for epoch in range(num_epochs):
        for i, (data, labels) in enumerate(train_dl):
            optimizer.zero_grad() 
            x_tensor = torch.tensor(data).to(device)
            y_tensor = torch.tensor([get_label(label) for label in labels]).to(device)
            # Forward pass; if IGNORE skip word, else feed model with embedding
            outputs = model(x_tensor)#(batchsize, sequencesize, vectorsize)
            loss = criterion(outputs, y_tensor.unsqueeze(1).float())

            # Backward and optimize
            
            loss.backward()
            optimizer.step()
            if (i+1) % 300 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        test(model,dev_dl,device, epoch)

def test(model,dev_dl,device, epoch):
    # Test the model \w (dev_loader)
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        y_true =[]
        y_pred = []
        for data, labels in dev_dl:
            x_tensor = torch.tensor(data).to(device)
            labels = np.array([get_label(label) for label in labels])
            outputs = model(x_tensor)
            # max returns (value ,index)
            predicted = outputs.cpu().detach().numpy()
            predicted = np.round(predicted)
            #labels = labels.float()
            y_true.extend(labels.tolist()) 
            y_pred.extend(predicted.reshape(-1).tolist())
        y_pred = [int(item) for item in y_pred]
        cm = confusion_matrix(y_true, y_pred)
        print(f'True Positive:\t{cm[0,0]}\tFalse Positve:\t{cm[0,1]}')
        print(f'False Negative:\t{cm[1,0]}\tTrue Negative:\t{cm[1,1]}')
        
        print(f'Accuracy of the network on the 10000 test images: {accuracy_score(y_true,y_pred)} and f1 score: {f1_score(y_true,y_pred)} %')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #Data Loaders
    batch_size = 2
    embed_size = 400
    hidden_size = 400
    num_layers = 2
    vocab_size = 14297
    learning_rate = 0.0005
    num_classes = 1
    num_epochs = 20
    dropout = 0.2

    #load data and create dataloaders
    train_data, dev_data, test_data = read_tokenized_data()
    train_dl = torch.utils.data.DataLoader(dataset=train_data, collate_fn=padding_collate_fn, batch_size=batch_size, shuffle=True)
    dev_dl = torch.utils.data.DataLoader(dataset=dev_data, collate_fn=padding_collate_fn, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(dataset=test_data, collate_fn=padding_collate_fn, batch_size=batch_size, shuffle=True)

    model = LSTM(embed_size, hidden_size, vocab_size, num_layers, num_classes, dropout = dropout).to(device)
    
    #Hyperparameters
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())  

   # retarded_data = train_data[:10]
    #retarded_dl = torch.utils.data.DataLoader(dataset=retarded_data, collate_fn=padding_collate_fn, batch_size=2, shuffle=True)
    #train(model,retarded_dl,retarded_dl,device,criterion,optimizer, 1000)
    print('training')
    train(model,train_dl,dev_dl,device,criterion,optimizer, num_epochs)


    # Test the model \w (dev_loader)
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        y_true =[]
        y_pred = []
        for data, labels in test_dl:
            x_tensor = torch.tensor(data).to(device)
            labels = np.array([get_label(label) for label in labels])
            outputs = model(x_tensor)
            # max returns (value ,index)
            predicted = outputs.cpu().detach().numpy()
            predicted = np.round(predicted)
            #labels = labels.float()
            y_true.extend(labels.tolist()) 
            y_pred.extend(predicted.reshape(-1).tolist())
        y_pred = [int(item) for item in y_pred]
        cm = confusion_matrix(y_true, y_pred)
        print(f'True Positive:\t{cm[0,0]}\tFalse Positve:\t{cm[0,1]}')
        print(f'False Negative:\t{cm[1,0]}\tTrue Negative:\t{cm[1,1]}')
        print(y_pred)
        print(y_true)
        print(f'Accuracy of the network on the 10000 test images: {accuracy_score(y_true,y_pred)} and f1 score: {f1_score(y_true,y_pred)} %')


