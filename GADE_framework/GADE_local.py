import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer, BertConfig, BertModel, DistilBertTokenizer, DistilBertConfig, DistilBertModel, RobertaTokenizer, RobertaConfig, RobertaModel

class GADE_local(nn.Module):
    def __init__(self, max_seq_length=128, device=0):
        super(GADE_local, self).__init__()
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
            nn.Linear(2 * self.dim, self.dim),
            nn.ReLU(),
            nn.Linear(self.dim, 1)
        )
        self.mlp = nn.Sequential(nn.Linear(768, 768),
                                 nn.PReLU(768),
                                 nn.Linear(768, 2))

    def encode_feature(self, cand_docs):
        input_ids = []
        segment_ids = []
        input_masks = []
        # for s in input_tokens:
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

    def forward(self, batch_data):
        features = []
        label = []

        for bd in batch_data:
            feat = self.encode_feature(bd)
            features.append(feat)
            label.append(bd["labels"])

        features = torch.stack(tuple(features), dim=0).to(self.device)
        features = features.view(-1, features.size(-1))
        features = self.mlp(features)
        label = torch.Tensor(label).to(self.device)

        return features, label



