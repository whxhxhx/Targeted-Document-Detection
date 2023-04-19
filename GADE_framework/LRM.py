import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer, BertConfig, BertModel, DistilBertTokenizer, DistilBertConfig, DistilBertModel, RobertaTokenizer, RobertaConfig, RobertaModel



class LRM(nn.Module):
    def __init__(self, max_seq_length=128, device=0):
        super(LRM, self).__init__()
        if not isinstance(device, list):
            device = [device]
        # self.device = torch.device("cuda:{:d}".format(device[0]))
        self.device = torch.device("cuda")
        self.max_seq_length = max_seq_length

        self.encoder_pretrain_path = '../plms/bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.encoder_pretrain_path, do_lower_case=True)
        self.config = BertConfig.from_pretrained(self.encoder_pretrain_path)

        if torch.cuda.is_available() and len(device) > 1:
            self.model = nn.DataParallel(BertModel.from_pretrained(self.encoder_pretrain_path, config=self.config), device_ids=device)
        else:
            self.model = BertModel.from_pretrained(self.encoder_pretrain_path, config=self.config)

        for param in self.model.parameters():
            param.requires_grad = True

        self.dim = 768

        self.similarity_network = nn.Sequential(
            nn.Linear(2 * self.dim, self.dim//6),
            nn.Tanh(),
            nn.Linear(self.dim//6, self.dim//12),
            nn.Tanh(),
            nn.Linear(self.dim//12, 1)
        )

    def encode_feature(self, cand_docs):
        input_ids = []
        segment_ids = []
        input_masks = []

        for s in cand_docs["input_tokens"]:
            tokens = ["[CLS]"] + cand_docs["description_token"] + ["[SEP]"] + s + ["[SEP]"]
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            seg_pos = len(cand_docs["description_token"]) + 2
            seg_ids = [0] * seg_pos + [1] * (len(tokens) - seg_pos)
            mask = [1] * len(tokens)
            padding = [0] * (self.max_seq_length - len(tokens))
            tokens += padding
            input_ids.append(tokens)
            seg_ids += padding
            segment_ids.append(seg_ids)
            mask += padding
            input_masks.append(mask)

        input_ids = torch.LongTensor(input_ids).cuda()
        segment_ids = torch.LongTensor(segment_ids).cuda()
        input_masks = torch.LongTensor(input_masks).cuda()
        outputs, pooled_output = self.model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks)

        return pooled_output

    def document_interaction_graph_construction(self, cand_docs, max_n):
        features = self.encode_feature(cand_docs)
        num_nodes, fdim = features.shape
        N = num_nodes
        A_feat = torch.cat([features.repeat(1, N).view(N * N, -1), features.repeat(N, 1)], dim=1).view(N, -1, 2 * self.dim)
        A_feat = self.similarity_network(A_feat).squeeze(2)
        A_feat = F.softmax(A_feat, dim=1)
        A_ = torch.zeros((max_n, max_n), dtype=torch.float32).to(self.device)
        A_[:num_nodes, :num_nodes] = A_feat

        labels = cand_docs["labels"].copy()
        mask = [1] * len(cand_docs["labels"])

        labels += [-10] * (max_n - num_nodes)
        mask += [0] * (max_n - num_nodes)
        features = torch.cat([features, torch.zeros((max_n - num_nodes, fdim), dtype=torch.float32).to(self.device)], dim=0)

        return features, A_, labels, mask

    def forward(self, batch_data):
        features = []
        inter_strength_mat = []
        label = []
        mask = []

        max_n = 0

        for bd in batch_data:
            if len(bd["labels"]) > max_n:
                max_n = len(bd["labels"])

        for bd in batch_data:
            feat, _A, l, m = self.document_interaction_graph_construction(bd, max_n)
            features.append(feat)
            inter_strength_mat.append(_A)
            label.append(l)
            mask.append(m)

        features = torch.stack(tuple(features), dim=0).to(self.device)
        inter_strength_mat = torch.stack(tuple(inter_strength_mat), dim=0).to(self.device)
        label = torch.Tensor(label).to(self.device)
        mask = torch.Tensor(mask).to(self.device)

        return features, inter_strength_mat, label, mask



