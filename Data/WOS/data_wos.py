import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import json
from collections import defaultdict
import datasets

np.random.seed(7)

def delete_existing_data():
    if os.path.exists('value_dict.pt'):
        os.remove('value_dict.pt')
    if os.path.exists('slot.pt'):
        os.remove('slot.pt')
    if os.path.exists('path_list.pt'):
        os.remove('path_list.pt')
    if os.path.exists('depth2label.pt'):
        os.remove('depth2label.pt')
    if os.path.exists('WebOfScience_train.json'):
        os.remove('WebOfScience_train.json')
    if os.path.exists('WebOfScience_dev.json'):
        os.remove('WebOfScience_dev.json')
    if os.path.exists('WebOfScience_test.json'):
        os.remove('WebOfScience_test.json')

def build_new_data():
    # Dictionary that maps label name to label id.
    label_dict = {}
    slot2value = defaultdict(set)
    data = datasets.load_dataset('json', data_files='wos_total.json')['train']
    for l in data['doc_label']:
        if l[0] not in label_dict:
            label_dict[l[0]] = len(label_dict)
    for l in data['doc_label']:
        assert len(l) == 2
        if l[1] not in label_dict:
            label_dict[l[1]] = len(label_dict)
        slot2value[label_dict[l[0]]].add(label_dict[l[1]])

    value_dict = {i: v for v, i in label_dict.items()}
    # Dictionary that maps label id to label name.
    torch.save(value_dict, 'value_dict.pt')
    # Dictionary that contains sets for each parent class which contains all associated children classes.
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

    id = [i for i in range(len(data))]
    np_data = np.array(id)
    np.random.shuffle(id)
    np_data = np_data[id]
    train, test = train_test_split(np_data, test_size=0.2, random_state=0)
    train, val = train_test_split(train, test_size=0.2, random_state=0)
    train = train.tolist()
    val = val.tolist()
    test = test.tolist()
    with open('WebOfScience_train.json', 'w') as f:
        for i in train:
            line = json.dumps({'token': data[i]['doc_token'], 'label': [label_dict[i] for i in data[i]['doc_label']]})
            f.write(line + '\n')
    with open('WebOfScience_dev.json', 'w') as f:
        for i in val:
            line = json.dumps({'token': data[i]['doc_token'], 'label': [label_dict[i] for i in data[i]['doc_label']]})
            f.write(line + '\n')
    with open('WebOfScience_test.json', 'w') as f:
        for i in test:
            line = json.dumps({'token': data[i]['doc_token'], 'label': [label_dict[i] for i in data[i]['doc_label']]})
            f.write(line + '\n')

if __name__ == '__main__':
    delete_existing_data()
    build_new_data()