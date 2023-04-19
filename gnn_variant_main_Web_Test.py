import logging
import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import mean
import argparse
from utils.dataset_generation import *
from torch.utils.data import DataLoader
from logger import set_logger
from utils.utils import *
from pytorch_transformers import AdamW, WarmupLinearSchedule

from GADE_framework.LRM import LRM
from GADE_framework.GIM import GIM
from ablation.SGC import SGC
from ablation.GCNII import GCNII

# os.environ["CUDA_VISIBLE_DEVICE"] = "0, 1"

f1_list = []


def load_entities(path):
    entity_list = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            entity_list.append(line[1])

    return entity_list

def calculate_f1(scores, labels):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    pred = (scores > 0.5).astype('int')

    TP = np.sum((pred == 1) * (labels == 1))
    TN = np.sum((pred == 0) * (labels == 0))
    FP = np.sum((pred == 1) * (labels == 0))
    FN = np.sum((pred == 0) * (labels == 1))
    acc = (TP + TN) * 1.0 / (TP + TN + FN + FP)
    if TP == 0:
        p = r = f1 = 0.0
    else:
        p = TP * 1.0 / (TP + FP)
        r = TP * 1.0 / (TP + FN)
        f1 = 2 * p * r / (p + r)

    return p, r, f1, acc


def test_gnn(iter, gim, lrm, criterion, prefix='Test'):
    gim.eval()
    lrm.eval()

    scores = []
    labels = []

    for j, batch in enumerate(iter):
        with torch.no_grad():
            feature, A, label, masks = lrm(batch)
            masks = masks.view(-1)
            label = label.view(-1)[masks == 1].long()
            pred = gim(feature, A)
            pred = pred[masks == 1]
            loss = criterion(pred, label)
            pred = F.softmax(pred, dim=1)
            p, r, acc = accuracy(pred, label)
            print(
                '{}\t[{:d}/{:d}]\tLoss {:.3f}\tAccuracy {:.3f}\tPrecison {:.3f}\tRecall {:.3f}'.format(prefix, j + 1,
                                                                                                       len(iter), loss,
                                                                                                       acc,
                                                                                                       p, r))
            assert pred.shape[0] == label.shape[0]
            scores += pred[:, 1].detach().cpu().numpy().tolist()
            labels += label.detach().cpu().numpy().tolist()

    p, r, f1, acc = calculate_f1(scores, labels)
    print('{}\tPrecison {:.3f}\tRecall {:.3f}\tF1-score {:.3f}\tAccuracy {:.3f}'.format(prefix, p, r, f1, acc))

    return f1

def extract_target_data(data_path, description_path, ent_list):
    with open(description_path, 'r') as f:
        description_json = json.load(f)

    with open(data_path, 'r') as f:
        men_context_json = json.load(f)

    ent_des = {}
    men_doc_dict = {}
    labels = {}
    e2m = {}

    for ent in ent_list:
        ent_des.update({ent: description_json[ent]})
        labels.update({ent: []})
        men_doc_dict.update({ent: []})

        for name, doc_list in men_context_json[ent].items():
            e2m.update({ent:name})
            for doc_ctx in doc_list:
                labels[ent].append(doc_ctx["label"])
                doc_content = doc_ctx["left_context"] + name + doc_ctx["right_context"]
                men_doc_dict[ent].append(doc_content)

    return {'ent_des': ent_des, 'men_doc_dict': men_doc_dict, 'labels': labels, 'e2m': e2m}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--seed', default=28, type=int)
    parser.add_argument('--exp_dir', default="./saved_ckpt", type=str)
    parser.add_argument('--log_freq', default=5, type=int)
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--max_node', type=int, default=165)

    # Optimization args
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--embed_lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dropout', type=float,default=0.4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)

    # Data path args
    parser.add_argument('--entity_path', type=str, default='datasets/Web_Test/target_entities.txt')
    parser.add_argument('--data_path', type=str, default='datasets/Web_Test/TDD_dataset.json')
    parser.add_argument('--description_path', type=str, default='datasets/Web_Test/entity_desc.json')
    parser.add_argument('--model_name', default='SGC', type=str)

    # Device
    parser.add_argument('--gpu', type=int, default=[0,1], nargs='+')
    parser.add_argument('--gcn_layer', default=1, type=int)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--lamda', type=float, default=0.5)
    parser.add_argument('--variant', action='store_true', default=False)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    gcn_dim = 768
    kfold = args.kfold
    ent_list = load_entity_list(args.entity_path)

    params = args.__dict__
    with open(os.path.join(args.exp_dir, "training_params.txt"), 'w') as writer:
        writer.write(str(params))

    for i in range(kfold):
        # local relevance model
        lrm = LRM(max_seq_length=args.max_seq_length, device=args.gpu)
        # global interaction model
        if args.model_name == 'SGC':
            gim = SGC(dims=[gcn_dim] * (args.gcn_layer + 1), dropout=args.dropout)
        else:
            gim = GCNII([gcn_dim] * (args.gcn_layer + 1), dropout=args.dropout, lamda=args.lamda, alpha=args.alpha, variant=args.variant)

        tokenizer = lrm.tokenizer
        input_tokens, label_inputs, desc_tokens = generate_data(ent_list, args.description_path,
                                                                args.data_path, tokenizer,
                                                                args.max_seq_length)

        test_ent = yield_example(ent_list, input_tokens, label_inputs, desc_tokens)
        test_dataset = ComparisonDataset(test_ent)
        test_iter = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=collate_fn, shuffle=False)

        if args.model_name == 'SGC':
            args.checkpoint_path = './saved_ckpt/SGC_300'
        else:
            args.checkpoint_path = './saved_ckpt/GCNII_300'

        ckp_path = args.checkpoint_path + "/{}_best.pth".format(i)
        checkpoint = torch.load(ckp_path)
        lrm.load_state_dict(checkpoint["lrm"], strict=False)
        gim.load_state_dict(checkpoint["gim"], strict=False)
        lrm = lrm.to(lrm.device)
        gim = gim.to(lrm.device)
        criterion = nn.CrossEntropyLoss().to(lrm.device)

        print("load from {}".format(ckp_path))
        f1_score = test_gnn(iter=test_iter, gim=gim, lrm=lrm, prefix='Test',
                      criterion=criterion, test_step=i + 1)
        print("Test F1 score\tfold\t{:d}\t{:.4f}".format(i, f1_score))
        f1_list.append(f1_score)

    print("5 fold test f1-scores is {}".format(f1_list))
    print("The average f1 score of 5 fold cross validation is {}".format(mean(f1_list)))
