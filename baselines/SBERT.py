import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer, BertConfig, BertModel, DistilBertTokenizer, DistilBertConfig, DistilBertModel, RobertaTokenizer, RobertaConfig, RobertaModel


class SentenceBert(nn.Module):
    def __init__(self, max_seq_length=128, device=0):
        super(SentenceBert, self).__init__()
        if not isinstance(device, list):
            device = [device]
        # self.device = torch.device("cuda:{:d}".format(device[0]))
        self.device = torch.device("cuda")
        self.max_seq_length = max_seq_length

        self.encoder_pretrain_path = '../plms/bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.encoder_pretrain_path, do_lower_case=True)
        self.config = BertConfig.from_pretrained(self.encoder_pretrain_path)

        if torch.cuda.is_available() and len(device) > 1:
            self.model = nn.DataParallel(BertModel.from_pretrained(self.encoder_pretrain_path, config=self.config),
                                         device_ids=device)
        else:
            self.model = BertModel.from_pretrained(self.encoder_pretrain_path, config=self.config)

        for param in self.model.parameters():
            param.requires_grad = True

        self.dim = 768

    def get_feature(self, example):
        input_ids = []
        segment_ids = []
        input_masks = []
        labels = []
        for s in example["input_tokens"]:
            if len(s) > self.max_seq_length - 2:
                s = s[:self.max_seq_length - 2]
            tokens = ["[CLS]"] + s + ["[SEP]"]
            tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            mask = [1] * len(tokens)
            padding = [0] * (self.max_seq_length - len(tokens))
            tokens += padding
            input_ids.append(tokens)
            mask += padding
            input_masks.append(mask)

        desc = example["description_token"]
        if len(desc) > self.max_seq_length - 2:
            desc = desc[:self.max_seq_length - 2]

        tokens = ["[CLS]"] + desc + ["[SEP]"]
        input_ids_des = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask_des = [1] * len(input_ids_des)
        padding = [0] * (self.max_seq_length - len(input_ids_des))
        input_ids_des += padding
        # input_ids.append(tokens)
        input_mask_des += padding
        # input_masks.append(mask)

        input_ids = torch.LongTensor(input_ids).cuda()
        input_masks = torch.LongTensor(input_masks).cuda()
        input_ids_des = torch.LongTensor(input_ids_des).cuda()
        input_mask_des = torch.LongTensor(input_mask_des).cuda()
        _, pooled_output = self.model(input_ids=input_ids, attention_mask=input_masks)
        features = pooled_output
        _, pooled_output_des = self.model(input_ids_des.unsqueeze(0), attention_mask=input_mask_des.unsqueeze(0))
        
        return features, pooled_output_des

    
    def forward(self, batch, max_n):
        feature = []
        label = []
        desc_feat = []
        mask = []
        for ex in batch:
            f, des = self.get_feature(ex)
            l = ex["labels"]
            num, fdim = f.shape
            f = torch.cat([f, torch.zeros((max_n - num, fdim), dtype=torch.float32).to(self.device)], dim=0)
            m = [1] * num + [0] * (max_n - num)
            feature.append(f)
            desc_feat.append(des)
            label.append(l)
            mask.append(m)

        feature = torch.stack(tuple(feature), dim=0).to(self.device)
        desc_feat = torch.stack(tuple(desc_feat), dim=0).to(self.device)
        label = torch.Tensor(label).to(self.device)
        mask = torch.Tensor(mask).to(self.device)

        return feature, desc_feat, label, mask


class SBERT_TDD(nn.Module):
    def __init__(self, max_seq_length=128, device=0):
        super(SBERT_TDD, self).__init__()
        self.encoder = SentenceBert(max_seq_length=max_seq_length, device=device)
        self.fc = nn.Linear(768 * 3, 2)

    def forward(self, batch_data):
        max_n = 0
        for bd in batch_data:
            if len(bd["labels"]) > max_n:
                max_n = len(bd["labels"])

        feature, desc_feat, label, mask = self.encoder(batch_data, max_n)
        # print(desc_feat.shape)
        desc_feat = desc_feat.repeat(1, max_n, 1)
        # print(feature.shape)
        feature = torch.cat([feature, desc_feat, torch.abs(feature - desc_feat)], dim=2)
        feature = feature.view(-1, feature.size(-1))
        logit = self.fc(feature)

        return logit, label, mask
