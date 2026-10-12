# -*- coding: utf-8 -*-
import os

import numpy as np
from gensim.models import Word2Vec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_symptoms(symptom_file):
    with open(symptom_file, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def load_pre_symptoms(pre_symptoms_file):
    pre_symptoms = []
    with open(pre_symptoms_file, encoding='utf-8') as f:
        for line in f:
            indices = [int(idx) for idx in line.replace(',', ' ').replace('\t', ' ').split()]
            pre_symptoms.append(indices)
    return pre_symptoms

def build_corpus(pre_symptoms, symptom_list, indices=None):
    corpus = []
    rows = range(len(pre_symptoms)) if indices is None else indices
    for i in rows:
        seq = [symptom_list[idx] for idx in pre_symptoms[i] if 0 <= idx < len(symptom_list)]
        if len(seq) >= 2:
            corpus.append(seq)
    return corpus

def load_external_corpus(tokenized_file):
    corpus = []
    with open(tokenized_file, encoding='utf-8') as f:
        for line in f:
            toks = line.strip().split()
            if len(toks) >= 2:
                corpus.append(toks)
    return corpus

def train_word2vec_two_stage(external_corpus, domain_corpus, vector_size=100,
                             pretrain_epochs=50, finetune_epochs=20):
    model = Word2Vec(
        sentences=external_corpus,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        epochs=pretrain_epochs,
    )
    n_ext = len(model.wv)

    if domain_corpus:
        model.build_vocab(domain_corpus, update=True)
        model.train(
            domain_corpus,
            total_examples=model.corpus_count,
            epochs=finetune_epochs,
        )
    n_final = len(model.wv)
    return model

def generate_feature_matrix(model, symptom_list):
    feature_matrix = np.zeros((len(symptom_list), model.vector_size))
    found = []
    for idx, symptom in enumerate(symptom_list):
        if symptom in model.wv:
            v = model.wv[symptom]
            if not np.isnan(v).any():
                feature_matrix[idx] = v
                found.append(idx)
    if found and len(found) < len(symptom_list):
        mean_vec = feature_matrix[found].mean(axis=0)
        if np.isnan(mean_vec).any():
            mean_vec = np.zeros(model.vector_size)
        found_set = set(found)
        for idx in range(len(symptom_list)):
            if idx not in found_set:
                feature_matrix[idx] = mean_vec
    return feature_matrix, len(found)

if __name__ == "__main__":
    symptom_file = os.path.join(ROOT, "data_TCM", "symptom_contains.txt")
    pre_symptoms_file = os.path.join(ROOT, "data_TCM", "pre_symptoms.txt")
    split_file = os.path.join(ROOT, "data_TCM", "split_indices.npz")
    external_corpus_file = os.path.join(ROOT, "data", "external_corpus", "external_corpus_tokenized.txt")
    out_file = os.path.join(ROOT, "data_TCM", "symptom_features_w2v.npy")

    symptom_list = load_symptoms(symptom_file)

    pre_symptoms = load_pre_symptoms(pre_symptoms_file)

    sp = np.load(split_file)
    train_idx = sp['train_idx'].tolist()
    domain_corpus = build_corpus(pre_symptoms, symptom_list, indices=train_idx)

    external_corpus = load_external_corpus(external_corpus_file)

    model = train_word2vec_two_stage(external_corpus, domain_corpus, vector_size=100)

    feature_matrix, n_covered = generate_feature_matrix(model, symptom_list)

    np.save(out_file, feature_matrix)
