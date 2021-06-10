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

def convert_into_embeddings(data):
    embdata= []
    for sentences in data:
        sentence = []
        for words in sentences:
            sentence.append(embeddings[words])
        embdata.append(sentence)
    return embdata

def train(model, train_dl, num_epochs, lr, device):

    opt = torch.optim.Adam(params=model.parameters(), lr=lr)
    crit = nn.NLLLoss()
    # this gives errors atm
    # TODO figure out how to pass input to the model also check transformer_models (forward and init)
    for epoch in range(num_epochs):
        for batch_idx, (data, labels) in enumerate(train_dl):
            opt.zero_grad()
            embdata = convert_into_embeddings(data)
            # data = [embeddings.get_vector(word) for word in sentence for sentence in data]
            # print(embdata)
            x_tensor = torch.tensor(embdata, dtype=torch.float32).to(device)
            y_tensor = torch.tensor([get_label(label[0]) for label in labels]).to(device)
            # print(data)
            # print(len(x_tensor))
            # print(y_tensor)
            out = model(x_tensor)
            loss = crit(out, y_tensor)
            loss.backward()
            opt.step()




if __name__ == '__main__':
    nltk.download('punkt')
    batch_size = 1

    train_data, test_data = read_tokenized_data()
    train_dl = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_dl = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    unique_tokens = []
    #print(train_data[0])
    for idx, (tweet, label) in enumerate(train_dl):
        for i in range(len(tweet[0][0])):
            if tweet[0][0][i] not in unique_tokens:
                unique_tokens.append(tweet[0][0][i])
    print(len(unique_tokens))
    for idx, (tweet, label) in enumerate(test_dl):
        print(tweet[0][0])
        print(label)
        break
    print(len(train_dl))
    print(len(test_dl))

    # copied the paramters that the transformer requires
    """
    :param emb: Embedding dimension
    :param heads: nr. of attention heads
    :param depth: Number of transformer blocks
    :param seq_length: Expected maximum sequence length
    :param num_tokens: Number of tokens (usually words) in the vocabulary
    :param num_classes: Number of classes.
    :param max_pool: If true, use global max pooling in the last layer. If false, use global
                     average pooling.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # paramters might make so sense this way especially look into embed dims and max sequence length
    # TODO
    # we need to give the transformer:
    # emb = dimmension of every word embedding
    # num_tokens = how many word (different vectors) do we have in the embedings
    # seq_length = max number of words (vectors) in a tweet
    #
    # HYPERPARAMETERS
    embed_size = 100
    heads = 5
    depth = 3
    seq_length = 103
    num_tokens = 16000
    num_classes = 2
    # need to make sure that emb / heads is an int
    model = CTransformer(embed_size, heads, depth, seq_length, num_tokens, num_classes=num_classes)

    train(model, train_dl, num_epochs=1, lr=1, device=device)


