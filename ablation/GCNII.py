import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MeanAggregator(nn.Module):
    def __init__(self):
        super(MeanAggregator, self).__init__()

    def forward(self, features, A):
        x = torch.bmm(A, features)
        return x


class GraphConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, aggregator, residual=False, variant=False):
        super(GraphConvModule, self).__init__()
        self.variant = variant
        if self.variant:
            self.in_dim = 2 * in_dim
        else:
            self.in_dim = in_dim
        # self.in_dim = in_dim
        self.out_dim = out_dim
        self.residual = residual
        self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)
        # nn.init.constant_(self.bias, 0)
        self.aggregator = aggregator()
        self.layernorm = nn.LayerNorm(self.out_dim)

    def forward(self, features, A, h0, lamda, alpha, l):
        theta = math.log(lamda/l+1)
        agg_features = self.aggregator(features, A)
        if self.variant:
            support = torch.cat([agg_features, h0], 1)
            r = (1-alpha)*agg_features+alpha*h0
        else:
            support = (1-alpha)*agg_features+alpha*h0
            r = support
        out = theta * torch.einsum('bnd,df->bnf', (support, self.weight)) + (1 - theta) * r
        
        if self.residual:
            out = out + features
        out = F.relu(out)

        return out


class GCNII(nn.Module):
    def __init__(self, dims, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = []
        self.layers = len(dims) - 1
        self.dropout = nn.Dropout(dropout)
        self.lamda = lamda
        self.alpha = alpha

        for i in range(len(dims) - 1):
            self.convs.append(GraphConvModule(in_dim=dims[i], out_dim=dims[i+1], aggregator=MeanAggregator, variant=variant))
        self.convs = nn.ModuleList(self.convs)
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(dims[0], dims[-1]))
        self.fcs.append(nn.Linear(dims[-1], dims[-1]))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.layernorm = nn.LayerNorm(dims[-1])
        self.comb_layer = nn.Linear(dims[-1]+dims[0], dims[-1])
        self.f1 = nn.Linear(dims[0], dims[-1])
        self.f2 = nn.Linear(dims[-1], dims[-1])
        self.binary_classifier = nn.Linear(dims[-1], 2)
        self.classifier = nn.Sequential(nn.Linear(dims[-1], dims[-1]),
                                        nn.PReLU(dims[-1]),
                                        nn.Linear(dims[-1], 2))

    def forward(self, x, A):
        _layers = []
        x = self.dropout(x)
        x_loc = x.view(-1, x.size(-1))
        layer_inner = F.relu(self.fcs[0](x))
        _layers.append(layer_inner)
        for i, conv in enumerate(self.convs):
            layer_inner = self.dropout(layer_inner)
            layer_inner = conv(layer_inner, A, _layers[0], self.lamda, self.alpha, i+1)
            
        layer_inner = self.fcs[-1](layer_inner)
        out = layer_inner.size(-1)
        layer_inner = layer_inner.view(-1, out)
        layer_inner = self.dropout(layer_inner)
        layer_inner += x_loc
        layer_inner = self.layernorm(layer_inner)
        pred = self.classifier(layer_inner)

        return pred
