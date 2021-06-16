import torch.nn as nn
from transformer_model import *
from preprocess import *
from torch.utils.data import DataLoader
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt

def get_label(y):
    if y == 'OFF':
        return 1
    return 0


def get_acc(dl, model, device, dev=False):
    correct, wrong = 0, 0
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

    for batch_idx, (data, labels) in enumerate(dl):
        y = get_label(labels[0])
        x_tensor = data.to(device)
        out = model(x_tensor)

        out = torch.argmax(out)
        out = out.detach().cpu().numpy()

        if out == y:
            correct += 1
            if out == 0:
                true_neg += 1
            else:
                true_pos += 1
        else:
            wrong += 1
            if out == 1:
                false_neg += 1
            else:
                false_pos += 1
    if dev:
        print("Dev set accuracy")
        print(correct / (correct + wrong))
        return correct / (correct + wrong)
    else:
        print("Testing accuracy")
        print(correct / (correct + wrong))
        print("F1 score")
        print(true_pos / (true_pos + .5 * (false_pos + false_neg)))
        print('Confusion matrix', true_pos, false_pos, true_neg, false_neg)

def train(model, train_dl, dev_dl, test_dl, num_epochs, device):
    model.to(device)
    crit = nn.NLLLoss()
    acc_list = []
    max_acc = 0
    for epoch in range(num_epochs):
        for batch_idx, (data, labels) in enumerate(train_dl):
            model.opt.zero_grad()
            x_tensor = data.to(device)
            y_tensor = torch.tensor([get_label(label) for label in labels]).to(device)
            out = model(x_tensor)
            loss = crit(out, y_tensor)
            loss.backward()
            model.opt.step()
        print("EPOCH", epoch)
        accuracy_this_epoch = get_acc(dev_dl, model, device, dev=True)
        get_acc(test_dl, model, device, dev=False)

        if accuracy_this_epoch > max_acc:
            max_acc = accuracy_this_epoch
            save_dict = {
                'epoch': 0,
                'weights': model.state_dict(),
                'optimizer': model.opt.state_dict()
            }
            model.save_checkpoint(save_dict, save_path='Transformer.ckpt')

        acc_list.append(accuracy_this_epoch)

if __name__ == '__main__':
    nltk.download('punkt')
    batch_size = 1

    train_data, dev_data, test_data = read_tokenized_data()

    train_dl = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    dev_dl = torch.utils.data.DataLoader(dataset=dev_data, batch_size=1, shuffle=True)
    test_dl = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # HYPERPARAMETERS
    embed_size = 128
    heads = 8
    depth = 6
    seq_length = 103
    num_tokens = 14298
    num_classes = 2
    L2 = 0.0008
    lr = 0.0005

    model = CTransformer(embed_size, heads, depth, seq_length=seq_length, num_tokens=num_tokens,
                         num_classes=num_classes, lr=lr, L2=L2)

    train(model, train_dl, dev_dl, test_dl, num_epochs=20, device=device)


