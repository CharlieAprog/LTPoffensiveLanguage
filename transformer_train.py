import torch.nn as nn
from transformer_model import *
from preprocess import *
from BERT_data import *
from torch.utils.data import DataLoader
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms

def get_label(y):
    if y == 'OFF':
        return 1
    return 0

def train(model, train_dl, num_epochs, lr, device):
    model.to(device)
    opt = torch.optim.Adam(params=model.parameters(), lr=lr)
    crit = nn.NLLLoss()

    for epoch in range(num_epochs):
        for batch_idx, (data, labels) in enumerate(train_dl):
            opt.zero_grad()
            x_tensor = data.to(device)
            y_tensor = torch.tensor([get_label(label[0]) for label in labels]).to(device)
            out = model(x_tensor)
            loss = crit(out, y_tensor)
            loss.backward()
            opt.step()

if __name__ == '__main__':
    nltk.download('punkt')
    batch_size = 1

    train_data, dev_data, test_data = read_tokenized_data()

    train_dl = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    dev_dl = torch.utils.data.DataLoader(dataset=dev_data, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # we need to give the transformer:
    # emb = dimmension of every word embedding
    # num_tokens = how many word (different vectors) do we have in the embedings
    # seq_length = max number of words (vectors) in a tweet
    #
    # HYPERPARAMETERS
    embed_size = 200
    heads = 5
    depth = 3
    seq_length = 103
    num_tokens = 14297
    num_classes = 2
    # need to make sure that emb / heads is an int
    # emb, heads, depth, seq_length, num_tokens, num_classes, max_pool = True, dropout = 0.0,
    model = CTransformer(embed_size, heads, depth, seq_length=seq_length, num_tokens=num_tokens, num_classes=num_classes)

    #train(model, train_dl, num_epochs=1, lr=1, device=device)


