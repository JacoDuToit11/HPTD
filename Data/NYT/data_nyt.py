import os
import xml.dom.minidom
from tqdm import tqdm
import json
import re
import tarfile
import shutil
from transformers import AutoTokenizer
from collections import defaultdict
import torch

"""
NYTimes Reference: https://catalog.ldc.upenn.edu/LDC2008T19
"""

sample_ratio = 0.02
train_ratio = 0.7
min_per_node = 200

source = []
labels = []
label_ids = []
label_dict = {}
sentence_ids = []
slot2value = defaultdict(set)
ROOT_DIR = 'Nytimes/'
label_f = 'nyt_label.vocab'

# 2003-07
def read_nyt(id_json):
    f = open(id_json, 'r')
    ids = f.readlines()
    f.close()
    print(ids[:2])
    f = open(label_f, 'r')
    label_vocab_s = f.readlines()
    f.close()
    label_vocab = []
    for label in label_vocab_s:
        label = label.strip()
        label_vocab.append(label)
    id_list = []
    for i in ids:
        id_list.append(int(i[13:-5]))
    print(id_list[:2])
    corpus = []

    for file_name in tqdm(ids):
        xml_path = file_name.strip()
        try:
            sample = {}
            dom = xml.dom.minidom.parse(xml_path)
            root = dom.documentElement
            tags = root.getElementsByTagName('p')
            text = ''
            for tag in tags[1:]:
                text += tag.firstChild.data
            if text == '':
                continue
            source.append(text)
            sample_label = []
            tags = root.getElementsByTagName('classifier')
            for tag in tags:
                type = tag.getAttribute('type')
                if type != 'taxonomic_classifier':
                    continue
                hier_path = tag.firstChild.data
                hier_list = hier_path.split('/')
                if len(hier_list) < 3:
                    continue
                for l in range(1, len(hier_list) + 1):
                    label = '/'.join(hier_list[:l])
                    if label == 'Top':
                        continue
                    if label not in sample_label and label in label_vocab:
                        sample_label.append(label)
            labels.append(sample_label)
            sentence_ids.append(file_name)
            sample['doc_topic'] = []
            sample['doc_keyword'] = []
            corpus.append(sample)
        except AssertionError:
            print(xml_path)
            print('Something went wrong...')
            continue

if __name__ == '__main__':
    # files = os.listdir('nyt_corpus/data')
    # for year in files:
    #     month = os.listdir(os.path.join('nyt_corpus/data', year))
    #     for m in month:
    #         if os.path.isdir(os.path.join('nyt_corpus/data', year, m)):
    #             new_month = os.listdir(os.path.join('nyt_corpus/data', year, m))
    #             for nm in new_month:
    #                 f = tarfile.open(os.path.join('nyt_corpus/data', year, m, nm))
    #                 f.extractall(os.path.join('Nytimes', year))
    #         else:
    #             f = tarfile.open(os.path.join('nyt_corpus/data', year, m))
    #             f.extractall(os.path.join('Nytimes', year))
    # files = os.listdir('Nytimes')
    # for year in files:
    #     month = os.listdir(os.path.join('Nytimes', year))
    #     for m in month:
    #         days = os.listdir(os.path.join('Nytimes', year, m))
    #         for d in days:
    #             file = os.listdir(os.path.join('Nytimes', year, m, d))
    #             for f in file:
    #                 shutil.move(os.path.join('Nytimes', year, m, d, f), os.path.join('Nytimes', year, f))
    read_nyt('idnewnyt_train.json')
    read_nyt('idnewnyt_val.json')
    read_nyt('idnewnyt_test.json')
    rev_dict = {}
    max_length = 0
    for l in labels:
        max_length = max(max_length, len(l))
        for l_ in l:
            split = l_.split('/')
            if l_ not in label_dict:
                label_dict[l_] = len(label_dict)
            for i in range(1, len(split) - 1):
                slot2value[label_dict['/'.join(split[:i + 1])]].add(label_dict['/'.join(split[:i + 2])])
                assert '/'.join(split[:i + 2]) not in rev_dict or rev_dict['/'.join(split[:i + 2])] == '/'.join(
                    split[:i + 1])
                rev_dict['/'.join(split[:i + 2])] = '/'.join(split[:i + 1])
    print(max_length)

    value_dict = {i: v.split('/')[-1] for v, i in
                  label_dict.items()}
    torch.save(value_dict, 'value_dict.pt')
    torch.save(slot2value, 'slot.pt')

    # Dictionary that maps each class to its parent class, if no parent class exists, -1 is assigned. 
    value2slot = {}

    num_class = 0
    for s in slot2value:
        for v in slot2value[s]:
            value2slot[v] = s
            if num_class < v:
                num_class = v
    num_class += 1

    # List of tuples that indicate the paths taken to leaf nodes.
    path_list = [(i, v) for v, i in value2slot.items()]

    for i in range(num_class):
        if i not in value2slot:
            value2slot[i] = -1

    # Obtains the depth of a class
    def get_depth(x):
        depth = 0
        while value2slot[x] != -1:
            depth += 1
            x = value2slot[x]
        return depth

    # Dictionary that maps each class to its depth in hierarchy.
    depth_dict = {i: get_depth(i) for i in range(num_class)}
    max_depth = depth_dict[max(depth_dict, key=depth_dict.get)] + 1
    # Dictionary that maps each level in hierarchy to a list of classes in that level.
    depth2label = {i: [a for a in depth_dict if depth_dict[a] == i] for i in range(max_depth)}

    # NOTE: this adds the virtual token thing used in the GAT at each level, i.e. it maps a 'level token' to each of the classes in that level.
    for depth in depth2label:
        for l in depth2label[depth]:
            path_list.append((num_class + depth, l))

    torch.save(path_list, 'path_list.pt')
    torch.save(depth2label, 'depth2label.pt')

    train_split = open('idnewnyt_train.json', 'r').readlines()
    dev_split = open('idnewnyt_val.json', 'r').readlines()
    test_split = open('idnewnyt_test.json', 'r').readlines()
    train, test, val = [], [], []
    for i in range(len(sentence_ids)):
        if sentence_ids[i] in train_split:
            train.append(i)
        elif sentence_ids[i] in dev_split:
            val.append(i)
        elif sentence_ids[i] in test_split:
            test.append(i)
        else:
            raise RuntimeError

    with open('NYT_train.json', 'w') as f:
        for i in train:
            line = json.dumps({'token': source[i], 'label': [label_dict[i] for i in labels[i]]})
            f.write(line + '\n')

    with open('NYT_dev.json', 'w') as f:
        for i in val:
            line = json.dumps({'token': source[i], 'label': [label_dict[i] for i in labels[i]]})
            f.write(line + '\n')

    with open('NYT_test.json', 'w') as f:
        for i in test:
            line = json.dumps({'token': source[i], 'label': [label_dict[i] for i in labels[i]]})
            f.write(line + '\n')
