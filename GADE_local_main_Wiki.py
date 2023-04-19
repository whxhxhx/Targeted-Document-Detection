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

from GADE_framework.GADE_local import GADE_local

# os.environ["CUDA_VISIBLE_DEVICE"] = "0, 1"

f1_list = []

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


def test(iter, logger, model, criterion, test_step=None, prefix='Test'):
    model.eval()

    scores = []
    labels = []

    for j, batch in enumerate(iter):
        with torch.no_grad():
            pred, label = model(batch)
            label = label.view(-1).long()
            loss = criterion(pred, label)
            pred = F.softmax(pred, dim=1)
            p, r, acc = accuracy(pred, label)
            logger.info(
                '{}\t[{:d}/{:d}]\tLoss {:.3f}\tAccuracy {:.3f}\tPrecison {:.3f}\tRecall {:.3f}'.format(prefix, j + 1,
                                                                                                       len(iter), loss,
                                                                                                       acc,
                                                                                                       p, r))
            assert pred.shape[0] == label.shape[0]
            scores += pred[:, 1].detach().cpu().numpy().tolist()
            labels += label.detach().cpu().numpy().tolist()

    p, r, f1, acc = calculate_f1(scores, labels)
    logger.info(
        '{}\tPrecison {:.3f}\tRecall {:.3f}\tF1-score {:.3f}\tAccuracy {:.3f}'.format(prefix, p, r, f1, acc))

    return f1


def train(iter, checkpoint_path, logger, fold, model, optimizer, criterion, epoch_num,
          start_epoch=0, scheduler=None, test_iter=None, val_iter=None, log_freq=1, start_f1=None):

    step = 0
    if start_f1 is None:
        best_f1 = 0.0
    else:
        best_f1 = start_f1

    for i in range(start_epoch, epoch_num):
        model.train()

        for j, batch in enumerate(iter):
            optimizer.zero_grad()
            step += 1
            pred, label = model(batch)
            label = label.view(-1).long()
            loss = criterion(pred, label)
            p, r, acc = accuracy(pred, label)
            # optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            if scheduler:
                scheduler.step()

            if (j + 1) % log_freq == 0:
                logger.info(
                    'Train\tEpoch:[{:d}][{:d}/{:d}]\tLoss {:.3f}\tAccuracy {:.3f}\tPrecison {:.3f}\tRecall {:.3f}'.format(
                        i, j + 1, len(iter), loss, acc, p, r))

        if val_iter:
            f1_score = test(iter=val_iter, logger=logger, model=model, prefix='Val',
                      criterion=criterion, test_step=i + 1)
            if f1_score > best_f1:
                best_f1 = f1_score
                state = {
                    "model": model.state_dict(),
                    "epoch": i+1,
                    "val_f1": best_f1
                }
                torch.save(state, os.path.join(checkpoint_path, "{}_best.pth".format(fold)))
                logger.info("Val Best F1-score\t{:.4f}".format(best_f1))

    if test_iter:
        checkpoint = torch.load(os.path.join(checkpoint_path, "{}_best.pth".format(fold)))
        model.load_state_dict(checkpoint["model"])
        model = model.to(model.device)
        best_epoch = checkpoint["epoch"]
        val_f1 = checkpoint["val_f1"]
        logger.info("load from epoch {:d}  f1 score {:.4f}".format(best_epoch, val_f1))
        f1_score = test(iter=test_iter, logger=logger, model=model, prefix='Test',
                      criterion=criterion, test_step=i + 1)
        logger.info("Test F1 score\tEpoch\t{:d}\t{:.4f}".format(best_epoch, f1_score))
        f1_list.append(f1_score)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    working_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--seed', default=28, type=int)
    parser.add_argument('--exp_dir', default=".", type=str)
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
    parser.add_argument('--checkpoint_path', default="./saved_ckpt", type=str)
    parser.add_argument('--data_type', type=str, default='Wiki300')
    parser.add_argument('--model_name', default='GADE_local_300', type=str)

    # Device
    parser.add_argument('--gpu', type=int, default=[0,1], nargs='+')
    parser.add_argument('--gcn_layer', default=1, type=int)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    gcn_dim = 768
    kfold = args.kfold

    params = args.__dict__
    with open(os.path.join(args.exp_dir, "training_params.txt"), 'w') as writer:
        writer.write(str(params))

    args.entity_path = 'datasets/' + args.data_type + '/target_entities.txt'
    args.data_path = 'datasets/' + args.data_type + '/TDD_dataset.json'
    args.description_path = 'datasets/' + args.data_type + '/entity_desc.json'
    ent_list = load_entity_list(args.entity_path)

    for i in range(kfold):
        model = GADE_local(max_seq_length=args.max_seq_length, device=args.gpu)
        tokenizer = model.tokenizer
        input_tokens, label_inputs, desc_tokens = generate_data(ent_list, args.description_path,
                                                                args.data_path, tokenizer,
                                                                args.max_seq_length)
        train_ent, val_ent, test_ent = get_kfold_data(ent_list, kfold, i)

        train_ent = yield_example(train_ent, input_tokens, label_inputs, desc_tokens)
        val_ent = yield_example(val_ent, input_tokens, label_inputs, desc_tokens)
        test_ent = yield_example(test_ent, input_tokens, label_inputs, desc_tokens)
        train_dataset = ComparisonDataset(train_ent)
        val_dataset = ComparisonDataset(val_ent)
        test_dataset = ComparisonDataset(test_ent)
        train_iter = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
        val_iter = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)
        test_iter = DataLoader(test_dataset, batch_size=args.test_batch_size, collate_fn=collate_fn, shuffle=False)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.embed_lr},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.embed_lr},
        ]

        num_train_steps = len(train_iter) * args.epochs
        opt = AdamW(optimizer_grouped_parameters, eps=1e-8)
        scheduler = WarmupLinearSchedule(opt, warmup_steps=0, t_total=num_train_steps)

        if not os.path.exists(args.checkpoint_path):
            os.makedirs(args.checkpoint_path)

        checkpoint_path = args.checkpoint_path + '/' + args.model_name
        log_dir = os.path.join(args.exp_dir, "logs")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        logger = set_logger(os.path.join(log_dir, str(time.time()) + "_" + args.model_name + ".log"))

        start_epoch = 0
        start_f1 = 0.0


        model = model.to(model.device)
        criterion = nn.CrossEntropyLoss().to(model.device)

        logger.info("The {}-th fold training begins!".format(i))
        train(train_iter, checkpoint_path, logger, i, model, opt, criterion, args.epochs, test_iter=test_iter,
              val_iter=val_iter, scheduler=scheduler, log_freq=args.log_freq, start_epoch=start_epoch, start_f1=start_f1)

    logger.info("5 fold test f1-scores is {}".format(f1_list))
    logger.info("The average f1 score of 5 fold cross validation is {}".format(mean(f1_list)))
