import argparse
import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from baselines.bm25 import *
from baselines.textrank_bm25 import *
from baselines.rake_bm25 import *
import torch


def load_entity_list(path):
    entity_list = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')
            entity_list.append(line[1])
    return entity_list


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


def get_fold_data(data_path, description_path, ent_list, kfold, sample_fold):
    num = len(ent_list)
    fold_size = num // kfold
    train_ent = None
    val_ent = None
    test_ent = None
    for j in range(kfold):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        data_part = ent_list[idx]
        if j == sample_fold:
            test_ent = data_part
            
        elif train_ent is None:
            train_ent = data_part
        else:
            train_ent = train_ent + data_part
    val_ent = train_ent[:fold_size]
    train_ent = train_ent[fold_size:]

    val_dataset = extract_target_data(data_path, description_path, val_ent)
    test_dataset = extract_target_data(data_path, description_path, test_ent)

    return val_dataset, test_dataset


def calculate_f1(scores, labels, thredhold):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    pred = (scores > thredhold).astype('int')

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


def validate(model, dataset, thredhold_list):
    ent_des, men_doc_dict, labels, e2m = dataset['ent_des'], dataset['men_doc_dict'], dataset['labels'], dataset['e2m']
    scores = []
    total_label = []
    for e, des in ent_des.items():
        target_label = labels[e]
        doc_list = men_doc_dict[e]
        m = e2m[e]
        sim_score = model.run(des, doc_list, target_label, m)
        if not isinstance(sim_score, list):
            sim_score = torch.sigmoid(sim_score)
            sim_score = sim_score.detach().tolist()
        scores.extend(sim_score)
        total_label.extend(target_label)
    
    best_thred = 0.0
    best_val_f1 = 0.0
    for thredhold in thredhold_list:
        p, r, f1, acc = calculate_f1(scores, total_label, thredhold)
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_thred = thredhold
    return best_val_f1, best_thred


def test(model, dataset, thredhold):
    ent_des, men_doc_dict, labels, e2m = dataset['ent_des'], dataset['men_doc_dict'], dataset['labels'], dataset['e2m']
    scores = []
    total_label = []
    for e, des in ent_des.items():
        target_label = labels[e]
        doc_list = men_doc_dict[e]
        m = e2m[e]
        sim_score = model.run(des, doc_list, target_label, m)
        if not isinstance(sim_score, list):
            sim_score = torch.sigmoid(sim_score)
            sim_score = sim_score.detach().tolist()
        scores.extend(sim_score)
        total_label.extend(target_label)

    p, r, f1, acc = calculate_f1(scores, total_label, thredhold)
    return f1


def test_tf_idf(dataset, thredhold):
    ent_des, men_doc_dict, labels, e2m = dataset['ent_des'], dataset['men_doc_dict'], dataset['labels'], dataset['e2m']
    scores = []
    total_label = []
    for e, des in ent_des.items():
        target_label = labels[e]
        doc_list = men_doc_dict[e]
        m = e2m[e]
        vectorizer = CountVectorizer(stop_words="english")
        tokenized_query = des.split(" ")
        tokenized_query = m.split(" ") + tokenized_query
        new_query = " ".join(tokenized_query)
        doc_list.insert(0, new_query)
        embeddings = vectorizer.fit_transform(doc_list)
        cos_sim = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()
        doc_scores = torch.Tensor(cos_sim)
        sim_scores = torch.sigmoid(doc_scores).detach().tolist()
        scores.extend(sim_scores)
        total_label.extend(target_label)

    p, r, f1, acc = calculate_f1(scores, total_label, thredhold)
    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data path args
    parser.add_argument('--model', type=str, default='tf_idf')
    parser.add_argument('--entity_path', type=str, default='datasets/wiki300/target_entities.txt')
    parser.add_argument('--data_path', type=str, default='datasets/wiki300/TDD_dataset.json')
    parser.add_argument('--description_path', type=str, default='datasets/wiki300/entity_desc.json')
    parser.add_argument('--data_type', type=str, default='wiki')

    args = parser.parse_args()

    kfold = 5
    ent_list = load_entity_list(args.entity_path)
    result_path = args.result_path

    results = []
    thred_list = []
    ave_f1 = 0.0
    for i in range(kfold):
        if args.data_type == 'wiki':
            val_dataset, test_dataset = get_fold_data(args.data_path, args.description_path, ent_list, kfold, i)
        else:
            test_dataset = extract_target_data(args.data_path, args.description_path, ent_list)
            
        if args.model == 'tf_idf':
            thredhold = 0.5
            test_f1 = test_tf_idf(test_dataset, thredhold)
        elif args.model == 'bm25':
            thredhold = 0.5
            model = BM25_Ranker()
            test_f1 = test(model, test_dataset, thredhold)
        elif args.model == 'textrank_bm25':
            thredhold = 0.1
            model = TextRank_BM25()
            test_f1 = test(model, test_dataset, thredhold)
        elif args.model == 'rake_bm25':
            thredhold = 0.1
            model = Rake_BM25()
            test_f1 = test(model, test_dataset, thredhold)

        results.append(test_f1)

    for i, r in enumerate(results):
        print(r)
        ave_f1 += r
    ave_f1 /= len(results)
    print('The average f1 of 5-fold is {}'.format(ave_f1))