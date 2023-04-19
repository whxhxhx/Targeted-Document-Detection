import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import BertTokenizer, BertConfig, BertModel, DistilBertTokenizer, DistilBertConfig, DistilBertModel
from torch.cuda.amp import autocast


class BiEncoder(nn.Module):
    def __init__(self, max_seq_length=128, device=0):
        super(BiEncoder, self).__init__()
        if not isinstance(device, list):
            device = [device]
        # self.device = torch.device("cuda:{:d}".format(device[0]))
        self.device = torch.device("cuda")
        self.max_seq_length = max_seq_length
        self.max_des_length = max_seq_length // 2
        self.max_ctx_length = max_seq_length - self.max_des_length
        self.device = torch.device("cuda")

        self.encoder_pretrain_path = 'plms/bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.encoder_pretrain_path, do_lower_case=True)
        self.config = BertConfig.from_pretrained(self.encoder_pretrain_path)

        if torch.cuda.is_available() and len(device) > 1:
            self.model = nn.DataParallel(BertModel.from_pretrained(self.encoder_pretrain_path, config=self.config), device_ids=device)
            self.des_model = nn.DataParallel(BertModel.from_pretrained(self.encoder_pretrain_path, config=self.config),
                                         device_ids=device)
            
        else:
            self.model = BertModel.from_pretrained(self.encoder_pretrain_path, config=self.config)
            self.des_model = BertModel.from_pretrained(self.encoder_pretrain_path, config=self.config)
            
        for param in self.model.parameters():
            param.requires_grad = True

        for param in self.des_model.parameters():
            param.requires_grad = False

        self.dim = 768

    def get_feature(self, example):
        
        input_ids = []
        segment_ids = []
        input_masks = []
        labels = []
        
        for s in example["input_tokens"]:
            if len(s) > self.max_seq_length - 2:
                s = s[:self.max_seq_length-2]
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
        tokens = self.tokenizer.convert_tokens_to_ids(tokens)
        mask = [1] * len(tokens)
        padding = [0] * (self.max_seq_length - len(tokens))
        tokens += padding
        input_ids.append(tokens)
        mask += padding
        input_masks.append(mask)

        input_ids = torch.LongTensor(input_ids).cuda()
        input_masks = torch.LongTensor(input_masks).cuda()
        _, pooled_output = self.model(input_ids=input_ids[:-1], attention_mask=input_masks[:-1])
        features = pooled_output
        _, pooled_output_des = self.des_model(input_ids[-1].unsqueeze(0), attention_mask=input_masks[-1].unsqueeze(0))
        score = torch.matmul(features, pooled_output_des.unsqueeze(-1)).squeeze(-1)
        score = torch.sigmoid(score)

        return score

    def forward(self, batch):
        feature = []
        label = []
        mask = []
        max_n = 0
        for bd in batch:
            if len(bd["labels"]) > max_n:
                max_n = len(bd["labels"])
        for ex in batch:
            f = self.get_feature(ex)
            num, fdim = f.shape
            f = torch.cat([f, torch.zeros((max_n - num, fdim), dtype=torch.float32).to(self.device)], dim=0)
            l = ex["labels"] + [-10] * (max_n - len(ex["label"]))
            m = [1] * len(ex["labels"]) + [0] * (max_n - len(ex["label"]))
            feature.append(f)
            label.append(l)
            mask.append(m)

        feature = torch.stack(tuple(feature), dim=0).to(self.device)
        label = torch.Tensor(label).to(self.device)
        mask = torch.Tensor(mask).to(self.device)
        feature = feature.view(-1)

        return feature, label, mask
