import numpy as np
from collections import Counter, OrderedDict
from typing import List, Dict
import random
import math
import copy
import torch
import pandas as pd 
from torch.distributions.dirichlet import Dirichlet
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, FashionMNIST
import torchvision.transforms as transforms
import torch.nn as nn
from sklearn.cluster import OPTICS
from models import fMLP, fCNN, eMLP, eCNN, CNN2, CNN4, LSTM, CustomCNN, ResNet101, ResNet50, VGG16, CustomDataset
from tqdm import tqdm 

from distance_utils import hellinger, jensen_shannon_divergence_distance



def renormalize(dist: torch.tensor, labels: List[int], label: int):
    idx = labels.index(label)
    dist[idx] = 0
    dist /= sum(dist)
    dist = torch.concat((dist[:idx], dist[idx+1:]))
    return dist

def load_data(dataset: str):
    if dataset == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = CIFAR10("data", train=True, download=True, transform=train_transform)
        testset = CIFAR10("data", train=False, download=True, transform=test_transform)
        datatype = 'image'

    elif dataset == "emnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        trainset = EMNIST("data", split="balanced", train=True, download=True, transform=transform)
        testset = EMNIST("data", split="balanced", train=False, download=True, transform=transform)

    elif dataset == "fmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        trainset = FashionMNIST(root='data', train=True, download=True, transform=transform)
        testset = FashionMNIST(root='data', train=True, download=True, transform=transform)

    elif dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = CIFAR100("data", train=True, download=True, transform=train_transform)
        testset = CIFAR100("data", train=False, download=True, transform=test_transform)

    elif dataset == 'sentimen140':
        import gdown
        from sklearn.model_selection import train_test_split
        import zipfile
        from pathlib import Path
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        file_path = Path("/dataset/training.1600000.processed.noemoticon.csv")

        if file_path.exists():
            print("Dataset is already dowloaded!")
        else:
            file_id = '1AN_svT4t-3U7otJjavvy-2SCXBX-NyBb'
            output_path = "dataset.zip"

            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

            with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
                zip_ref.extractall("dataset")

        columns = ['label', 'id', 'date', 'flag', 'user', 'text']
        df = pd.read_csv(file_path, names=columns, encoding='latin1')

        max_words = 2000
        max_len = 500

        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(df['text'])
        sequences = tokenizer.texts_to_sequences(df['text'])

        X = pad_sequences(sequences, maxlen=max_len)
        y = [0 if i == 0 else 1 for i in df['label'].values]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        trainset = CustomDataset(X_train, y_train)
        testset = CustomDataset(X_test, y_test)

    return trainset, testset

def clustering(classes, data_size, num_clients, dist, min_smp=3, xi=0.01, distance='manhattan'):
    distrib_ = []
    for d in dist:
      distrib_.append(np.array([d[s] / (data_size / num_clients) if d[s] is not None else 0 for s in classes]))

    if distance == 'hellinger':
        optics = OPTICS(min_samples=min_smp,
                        xi=xi,
                        metric=hellinger,
                        min_cluster_size=2)
    elif distance == 'jensenshannon':
        optics = OPTICS(min_samples=min_smp,
                        xi=xi,
                        metric=jensen_shannon_divergence_distance,
                        min_cluster_size=2)
    else:
        optics = OPTICS(min_samples=min_smp,
                        xi=xi,
                        metric=distance,
                        min_cluster_size=2)

    optics.fit(distrib_)
    labels = optics.labels_
    uniq_labels = torch.unique(torch.tensor(labels))

    client_cluster_index = dict()
    for i, lab in enumerate(labels):
        client_cluster_index[i] = int(lab + 1)

    return client_cluster_index, distrib_

def partition_data(dataset, _iid: int, non_iid_diff : int, num_clients: int, alpha: float, beta: float, dataset_name='cifar10'):
    assert _iid + non_iid_diff <= num_clients, 'Check num_iid, non_iid_diff and num_clients.'

    classes_ = dataset.classes
    num_classes = len(classes_)

    client_size = len(dataset) // num_clients
    label_size = len(dataset) // num_classes

    indices_class = [[] for _ in range(num_classes)]

    for i, lab in enumerate(dataset.targets):
        indices_class[lab].append(i)

    if dataset_name == 'sentimen140':
        non_iid_labels = list(range(2))
        id_non_iid_clients_size = client_size

    elif dataset_name == 'cifar10' or dataset_name == 'fmnist':
        non_iid_labels = random.sample(range(num_classes), 2)
        id_non_iid_clients_size = client_size

    elif dataset_name == 'emnist':
        non_iid_labels = list(range(10))
        id_non_iid_clients_size = client_size

    elif dataset_name == 'cifar100':
        non_iid_labels = random.sample(range(num_classes), 15)
        id_non_iid_clients_size = client_size

    non_iid_data = []
    labels = list(range(num_classes))

    for label in non_iid_labels:
        non_iid_data += indices_class[label]

    ids = []
    label_dist = []

    print('Processing non-iid and iid---')
    for i in tqdm(range(non_iid_diff + _iid)):
        concentration = torch.ones(len(labels)) * (alpha if i < _iid else beta)
        dist = Dirichlet(concentration).sample()

        client_indices = []
        for _ in range(client_size):
            if not labels:
                break

            label = random.choices(labels, dist)[0]
            if indices_class[label]:
                id_sample = random.choice(indices_class[label])
                client_indices.append(id_sample)
                indices_class[label].remove(id_sample)

                if not indices_class[label]:
                    dist = renormalize(dist, labels, label)
                    labels.remove(label)

        ids.append(client_indices)
        counter = Counter(list(map(lambda x: dataset[x][1], ids[i])))
        label_dist.append({classes_[j]: counter.get(j, 0) for j in range(num_classes)})
    
    print('Processing identical distributed non-iid')
    for i in tqdm(range(non_iid_diff + _iid, num_clients)):

        temp_data = non_iid_data.copy()
        id_sample = random.sample(temp_data, id_non_iid_clients_size)
        ids.append(id_sample)

        counter = Counter(list(map(lambda x: dataset[x][1], ids[i])))
        label_dist.append({classes_[j]: counter.get(j, 0) for j in range(num_classes)})

    return ids, label_dist

