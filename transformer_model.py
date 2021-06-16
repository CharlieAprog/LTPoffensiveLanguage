import torch
from torch import nn
import torch.nn.functional as F

from transformer_tools.modules import TransformerBlock
from transformer_tools.utils import *
from transformer_tools.utils_small import d

class CTransformer(nn.Module):
    """
    Transformer for classifying sequences
    """

    def __init__(self,emb,heads,depth,seq_length,num_tokens,num_classes,lr,L2,max_pool=True,dropout=0.0,wide=False):
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
        super().__init__()

        self.num_tokens, self.max_pool = num_tokens, max_pool
        # we might not need these embeddings at all
        # however it could be that transformers work best when they do their own embeds idk
        self.token_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=num_tokens)
        self.pos_embedding = nn.Embedding(embedding_dim=emb, num_embeddings=seq_length)

        tblocks = []
        for i in range(depth):
            tblocks.append(
                TransformerBlock(emb=emb, heads=heads, seq_length=seq_length, mask=False, dropout=dropout))

        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(emb, num_classes)

        self.do = nn.Dropout(dropout)

        self.opt = torch.optim.Adam(params=self.parameters(), lr=lr, weight_decay=L2)

    # function to create and load ckpt files
    def load_checkpoint(self, ckpt_path, map_location=None):
        ckpt = torch.load(ckpt_path, map_location=map_location)
        print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
        return ckpt

    def save_checkpoint(self, state, save_path):
        torch.save(state, save_path)

    def load_model(self, ckpt):
        self.epoch = ckpt['epoch']
        self.load_state_dict(ckpt['weights'])
        self.opt.load_state_dict(ckpt['optimizer'])

    def forward(self, x):
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        seqlen = [((x.size()[1] - (batch == 26).sum())) for batch in x]

        tokens = self.token_embedding(x)

        b, t, e = tokens.size()

        positions = self.pos_embedding(torch.arange(t, device=d()))[None, :, :].expand(b, t, e)

        x = tokens + positions
        # make every padding entry 0 before we feed it to the model
        for i in range(int(seqlen[0]), int(x.size()[1])):
            x[0][i] = torch.zeros([128])

        x = self.do(x)

        x = self.tblocks(x)

        x = x.max(dim=1)[0] if self.max_pool else x.mean(dim=1)  # pool over the time dimension

        x = self.toprobs(x)

        return F.log_softmax(x, dim=1)
