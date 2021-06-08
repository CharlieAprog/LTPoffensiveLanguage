import torch.nn as nn
from transformer_model import *
from preprocess import *
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
            print(embdata)
            x_tensor = torch.tensor(embdata, dtype=torch.float32).to(device)
            y_tensor = torch.tensor([get_label(label[0]) for label in labels]).to(device)
            print(data)
            print(len(x_tensor))
            print(y_tensor)
            out = model(x_tensor)
            loss = crit(out, y_tensor)
            loss.backward()
            opt.step()




if __name__ == '__main__':
    nltk.download('punkt')
    # load
    training_set, dev_set, test_set, vocab, embeddings = load_data()

    train_loader = torch.utils.data.DataLoader(dataset=training_set, collate_fn=padding_collate_fn,
                                               batch_size=batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dataset=dev_set, collate_fn=padding_collate_fn, batch_size=batch_size,
                                             shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, collate_fn=padding_collate_fn, batch_size=batch_size,
                                              shuffle=True)
    #example_data, example_targets = examples.next()

    #HYPERPARAMETERS
    embed_size = 100
    heads = 5
    depth = 3
    num_tokens = 16000
    num_classes = 2
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
    model = CTransformer(embed_size, heads, depth, 500, num_tokens, num_classes=num_classes)

    train(model, train_loader, num_epochs=1, lr=1, device=device)


