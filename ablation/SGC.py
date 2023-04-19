import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SGC(nn.Module):
    def __init__(self, nfeat, dims, nclass, dropout=0.0):
        super(SGC, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.w = nn.Parameter(torch.FloatTensor(nfeat, dims[-1]))
        self.b = nn.Parameter(torch.FloatTensor(dims[-1]))
        nn.init.xavier_uniform_(self.w)
        nn.init.constant_(self.b, 0)
        self.layernorm = nn.LayerNorm(dims[-1])
        self.classifier = nn.Sequential(nn.Linear(dims[-1], dims[-1]),
                                        nn.PReLU(dims[-1]),
                                        nn.Linear(dims[-1], nclass))

    def forward(self, features, A):
        features = self.dropout(features)
        features_loc = features.view(-1, features.size(-1))
        agg_features = torch.bmm(A, features)
        sum_embed = torch.einsum('bnd,df->bnf', (agg_features, self.w))
        sum_embed += self.b
        out = sum_embed.view(-1, sum_embed.size(-1))
        out = self.dropout(out)
        out += features_loc
        out = self.layernorm(out)
        pred = self.classifier(out)

        return pred