import torch
import torch.nn as nn
import torch.nn.functional as F


class Aggregator(nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, features, A):
        x = torch.bmm(A, features)
        return x


# GCN Layer
class GraphConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, aggregator):
        super(GraphConvLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggregator = aggregator()
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

        self.layernorm = nn.LayerNorm(self.out_dim)

    def forward(self, features, A):
        batch, node_num, d = features.shape
        assert d == self.in_dim
        agg_features = self.aggregator(features, A)
        out_features = torch.einsum('bnd,df->bnf', (agg_features, self.weight))
        out_features += self.bias
        out = F.relu(out_features)
        return out


class GIM(nn.Module):
    def __init__(self, dims, dropout=0.0):
        super(GIM, self).__init__()
        self.convs = []
        self.layers = len(dims) - 1
        self.dropout = nn.Dropout(dropout)

        for i in range(len(dims) - 1):
            self.convs.append(GraphConvLayer(dims[i], dims[i+1], Aggregator))
        self.convs = nn.ModuleList(self.convs)
        self.layernorm = nn.LayerNorm(dims[-1])
        self.classifier = nn.Sequential(nn.Linear(dims[-1], dims[-1]),
                                        nn.PReLU(dims[-1]),
                                        nn.Linear(dims[-1], 2))

    def forward(self, x, A):
        x = self.dropout(x)
        # x_loc = x
        x_loc = x.view(-1, x.size(-1))
        for conv in self.convs:
            x = conv(x, A)

        out = x.size(-1)
        x = x.view(-1, out)
        x = x + x_loc
        x = self.layernorm(x)
        pred = self.classifier(x)

        return pred



