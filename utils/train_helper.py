import numpy as np
from collections import Counter, OrderedDict
from typing import List, Dict
import random
import math
import copy
import gc
import torch
import torch.nn as nn
from torch.optim import SGD
from models import CNN, LSTM, ResNet, ResidualBlock, VGG16
from tqdm import tqdm 

seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(model_name, dataset_name, model_config):
    """
    Only modify the num_conv_block if you want to use CustomCNN model
    """

    if dataset_name == 'sentimen140':
        if model_name == 'lstm':
            return LSTM()
        else:
            raise ValueError(f"Invalid model name '{model_name}' for dataset '{dataset_name}'")

    elif dataset_name == 'emnist':
        if model_name == 'cnn':
            return CNN(num_layer=model_config['num_layer'], 
                       in_shape=model_config['in_shape'],
                       im_size=model_config['im_size'],
                       hidden=model_config['hidden'], 
                       out_shape=model_config['out_shape'])
        else:
            raise ValueError(f"Invalid model name '{model_name}' for dataset '{dataset_name}'")

    elif dataset_name == 'fmnist':
        if model_name == 'cnn':
            return CNN(num_layer=model_config['num_layer'], 
                       in_shape=model_config['in_shape'],
                       im_size=model_config['im_size'],
                       hidden=model_config['hidden'], 
                       out_shape=model_config['out_shape'])
        else:
            raise ValueError(f"Invalid model name '{model_name}' for dataset '{dataset_name}'")

    elif dataset_name == 'cifar10':
        if model_name == 'cnn':
            return CNN(num_layer=model_config['num_layer'], 
                       im_size=model_config['in_shape'],
                       hidden=model_config['hidden'], 
                       out_shape=model_config['out_shape'])
        elif model_name == 'resnet50':
            return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=model_config['out_shape'])
        elif model_name == 'resnet101':
            return ResNet(ResidualBlock, [3, 4, 23, 3], num_classes=model_config['out_shape'])
        elif model_name == 'vgg16':
            return VGG16(in_channels=model_config['in_shape'], num_classes=model_config['out_shape'])
        else:
            raise ValueError(f"Invalid model name '{model_name}' for dataset '{dataset_name}'")

    elif dataset_name == 'cifar100':
        if model_name == 'cnn':
            return CNN(num_layer=model_config['num_layer'], 
                       im_size=model_config['in_shape'],
                       hidden=model_config['hidden'], 
                       out_shape=model_config['out_shape'])
        elif model_name == 'resnet50':
            return ResNet(ResidualBlock, [3, 4, 6, 3], num_classes=model_config['out_shape'])
        elif model_name == 'resnet101':
            return ResNet(ResidualBlock, [3, 4, 23, 3], num_classes=model_config['out_shape'])
        elif model_name == 'vgg16':
            return VGG16(in_channels=model_config['in_shape'], num_classes=model_config['out_shape'])
        else:
            raise ValueError(f"Invalid model name '{model_name}' for dataset '{dataset_name}'")

    else:
        raise ValueError(f"Invalid dataset name '{dataset_name}'")


def train(net, 
          trainloader, 
          criterion, 
          optimizer, 
          device, 
          proximal_mu: float = None):
    
    net.train()
    running_loss, running_corrects, tot = 0.0, 0, 0

    global_params = copy.deepcopy(net).parameters()

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)

        loss = criterion(outputs, labels)

        if proximal_mu is not None:
            proximal_term = sum((local_weights - global_weights).norm(2) 
                                for local_weights, global_weights in zip(net.parameters(), global_params))
            loss += (proximal_mu / 2) * proximal_term

        loss.backward()
        optimizer.step()

        predicted = torch.argmax(outputs, dim=1)
        tot += images.shape[0] 

        running_corrects += torch.sum(predicted == labels).item()
        running_loss += loss.item() * images.shape[0]

        del images, labels, outputs, loss, predicted

    running_loss /= tot
    accuracy = running_corrects / tot

    del global_params, tot
    torch.cuda.empty_cache()
    gc.collect()

    return running_loss, accuracy


def test(net, testloader, device):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    corrects, total_loss = 0, 0.0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            predicted = torch.argmax(outputs, dim=1)
            corrects += torch.sum(predicted == labels).item()
            total_loss += loss.item() * images.shape[0]

            del images, labels, outputs, predicted
            torch.cuda.empty_cache()

    total_loss /= len(testloader.dataset)
    accuracy = corrects / len(testloader.dataset)

    torch.cuda.empty_cache()
    gc.collect()

    return total_loss, accuracy



def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict)


def compute_entropy(counts: Dict):
    entropy = 0.0
    counts = list(counts.values())
    counts = [0 if value is None else value for value in counts]
    for value in counts:
        entropy += -value/sum(counts) * math.log(value/sum(counts), len(counts)) if value != 0 else 0
    return entropy
