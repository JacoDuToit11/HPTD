import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import xml.dom.minidom
import pandas as pd
import json
from collections import defaultdict

np.random.seed(7)

if __name__ == '__main__':
    source = []
    labels = []
    label_dict = {}
    slot2value = defaultdict(set)
    with open('rcv1.taxonomy', 'r') as f:
        label_dict['Root'] = -1
        for line in f.readlines():
            line = line.strip().split('\t')
            for i in line[1:]:
                if i not in label_dict:
                    label_dict[i] = len(label_dict) - 1
                slot2value[label_dict[line[0]]].add(label_dict[i])
        label_dict.pop('Root')
        slot2value.pop(-1)

    value_dict = {i: v for v, i in label_dict.items()}
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

    data = pd.read_csv('rcv1_v2.csv')
    for i, line in tqdm(data.iterrows()):
        dom = xml.dom.minidom.parseString(line['text'])
        root = dom.documentElement
        tags = root.getElementsByTagName('p')
        text = ''
        for tag in tags:
            text += tag.firstChild.data
        if text == '':
            continue
        source.append(text)
        l = line['topics'].split('\'')
        labels.append([label_dict[i] for i in l[1::2]])
    print(len(labels))

    # This is doing the train and test split by checking the specific ids of the instances.
    data = pd.read_csv('rcv1_v2.csv')
    ids = []
    for i, line in tqdm(data.iterrows()):
        dom = xml.dom.minidom.parseString(line['text'])
        root = dom.documentElement
        tags = root.getElementsByTagName('p')
        cont = False
        for tag in tags:
            if tag.firstChild.data != '':
                cont = True
                break
        if cont:
            ids.append(line['id'])

    train_ids = []
    with open('lyrl2004_tokens_train.dat', 'r') as f:
        for line in f.readlines():
            if line.startswith('.I'):
                train_ids.append(int(line[3:-1]))

    train_ids = set(train_ids)
    train = []
    test = []
    for i in range(len(ids)):
        if ids[i] in train_ids:
            train.append(i)
        else:
            test.append(i)

    id = [i for i in range(len(train))]
    np_data = np.array(train)
    np.random.shuffle(id)
    np_data = np_data[id]
    train, val = train_test_split(train, test_size=0.1, random_state=0)

    with open('RCV1_train.json', 'w') as f:
        for i in train:
            line = json.dumps({'token': source[i], 'label': labels[i]})
            f.write(line + '\n')
    with open('RCV1_dev.json', 'w') as f:
        for i in val:
            line = json.dumps({'token': source[i], 'label': labels[i]})
            f.write(line + '\n')
    with open('RCV1_test.json', 'w') as f:
        for i in test:
            line = json.dumps({'token': source[i], 'label': labels[i]})
            f.write(line + '\n')