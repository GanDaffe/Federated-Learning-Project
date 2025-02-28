import numpy as np
from collections import Counter, OrderedDict
from typing import List, Dict
import random
import math
import copy
import gc
import torch
from torch.distributions.dirichlet import Dirichlet
from torchvision.datasets import CIFAR10, EMNIST, FashionMNIST
import torchvision.transforms as transforms
import torch.nn as nn
from torch.optim import SGD
from scipy.spatial.distance import cosine
from sklearn.cluster import OPTICS
from models import fMLP, fCNN, eMLP, eCNN, CNN2, CNN4, LSTM, CustomCNN, ResNet101, ResNet50, VGG16
from tqdm import tqdm 

seed_value = 42
random.seed(seed_value)
torch.manual_seed(seed_value)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed_value)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_c_local(partition_id: int):
    path = "c_local_folder/" + str(partition_id) +".txt"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            c_delta_bytes = f.read()

        array = np.frombuffer(c_delta_bytes, dtype=np.float64)
        return array
    else:
        return None

# Custom function to serialize to bytes and save c_local variable inside a file
def set_c_local(partition_id: int, c_local):
    path = "training_process_files/c_local_folder/" + str(partition_id) +".txt"

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    c_local_list = []
    for param in c_local:
        c_local_list += param.flatten().tolist()

    c_local_numpy = np.array(c_local_list, dtype=np.float64)
    c_local_bytes = c_local_numpy.tobytes()

    with open(path, 'wb') as f:
        f.write(c_local_bytes)


def get_model(model_name, dataset_name, num_conv_block=4):
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
            return eCNN()
        elif model_name == 'mlp':
            return eMLP()
        else:
            raise ValueError(f"Invalid model name '{model_name}' for dataset '{dataset_name}'")

    elif dataset_name == 'fmnist':
        if model_name == 'cnn':
            return fCNN()
        elif model_name == 'mlp':
            return fMLP()
        else:
            raise ValueError(f"Invalid model name '{model_name}' for dataset '{dataset_name}'")

    elif dataset_name == 'cifar10':
        if model_name == 'cnn':
            return CustomCNN(in_shape=3, num_layer=num_conv_block, im_size=32, hidden=32, out_shape=10)

        elif model_name == 'resnet50':
            return ResNet50(num_classes=10)
        elif model_name == 'resnet101':
            return ResNet101(in_channel=3, num_classes=10)
        elif model_name == 'vgg16':
            return VGG16(in_channels=3, num_classes=10)
        else:
            raise ValueError(f"Invalid model name '{model_name}' for dataset '{dataset_name}'")

    elif dataset_name == 'cifar100':
        if model_name == 'cnn':
            return CustomCNN(in_shape=3, num_layer=num_conv_block, im_size=32, hidden=32, out_shape=100)
        elif model_name == 'resnet50':
            return ResNet50()
        elif model_name == 'resnet101':
            return ResNet101(in_channel=3, num_classes=100)
        elif model_name == 'vgg16':
            return VGG16(in_channels=3, num_classes=100)
        else:
            raise ValueError(f"Invalid model name '{model_name}' for dataset '{dataset_name}'")

    else:
        raise ValueError(f"Invalid dataset name '{dataset_name}'")


def train(net, trainloader, learning_rate: float, loss_fn, proximal_mu: float = None):

 
    criterion = nn.CrossEntropyLoss() if loss_fn == 'multiclass' else nn.BCELoss()
    optimizer = SGD(net.parameters(), lr=learning_rate)
    net.train()
    running_loss, running_corrects = 0.0, 0
    global_params = copy.deepcopy(net).parameters()

    for images, labels in trainloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = net(images).to(DEVICE)

        if proximal_mu != None:
            proximal_term = 0.0
            for local_weights, global_weights in zip(net.parameters(), global_params):
                proximal_term += (local_weights - global_weights).norm(2)
            loss = criterion(outputs, labels) + (proximal_mu / 2) * proximal_term
        else:
            loss = criterion(outputs, labels)
      
        loss.backward()
        optimizer.step()
        
        if loss_fn == 'multiclass':
            predicted = torch.argmax(outputs, dim=1)
        else:  # binary
            predicted = (outputs > 0.5).float()

        running_loss += loss.item() * images.shape[0]
        running_corrects += torch.sum(predicted == labels).item()

        if proximal_mu != None:
            del proximal_term
        del images, labels, outputs, loss, predicted
        torch.cuda.empty_cache()
        gc.collect()

    running_loss /= len(trainloader.sampler)
    accuracy = running_corrects / len(trainloader.sampler)

    del global_params
    torch.cuda.empty_cache()
    gc.collect()

    return running_loss, accuracy


def test(net, testloader, loss_fn):

    if loss_fn == 'multiclass':
        criterion = nn.CrossEntropyLoss()
    elif loss_fn == 'binary':
        criterion = nn.BCELoss()

    corrects, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images).to(DEVICE)
            predicted = torch.argmax(outputs, dim=1)
            loss += criterion(outputs, labels).item() * images.shape[0]

            if loss_fn == 'multiclass':
                predicted = torch.argmax(outputs, dim=1)
            else:  # binary
                predicted = (outputs > 0.5).float()
            
                
            corrects += torch.sum(predicted == labels).item()

            del images, labels, outputs, predicted
            torch.cuda.empty_cache()
            gc.collect()

    loss /= len(testloader.sampler)
    accuracy = corrects / len(testloader.sampler)

    torch.cuda.empty_cache()
    gc.collect()

    return loss, accuracy



def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict)


def train_scaffold(
    net: nn.Module,
    trainloader,
    optimizer,
    epochs: int,
    learning_rate: float,
    loss_fn: str,
    server_covariates: list,
    client_covariates: list,
):
    criterion = nn.CrossEntropyLoss() if loss_fn == 'multiclass' else nn.BCELoss()
    device = next(net.parameters()).device

    total_loss, total_corrects, total_samples = 0.0, 0, 0

    for epoch in range(epochs):
        net.train()
        running_loss, running_corrects = 0.0, 0

        for data, labels in trainloader:
            data, labels = data.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if loss_fn == 'multiclass':
                predicted = torch.argmax(outputs, dim=1)
                running_corrects += torch.sum(predicted == labels).item()
            else:  # binary
                predicted = (outputs > 0.5).float()
                running_corrects += torch.sum(predicted == labels).item()

            running_loss += loss.item() * data.size(0)  # Chuẩn hóa theo batch size
            total_samples += data.size(0)

            del outputs, loss, predicted
            if device.type == 'cuda':
                torch.cuda.empty_cache()

            weights = get_parameters(net)
            assert len(weights) == len(client_covariates) == len(server_covariates), \
                "Length mismatch between weights and covariates"
            for i in range(len(weights)):
                weights[i] -= learning_rate * (server_covariates[i] - client_covariates[i])
            set_parameters(net, weights)
            del weights

        total_loss += running_loss
        total_corrects += running_corrects

    # Giải phóng bộ nhớ sau khi hoàn tất
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    # Trả về trung bình toàn bộ quá trình
    return total_loss / total_samples, total_corrects / total_samples

def compute_entropy(counts: Dict):
    entropy = 0.0
    counts = list(counts.values())
    counts = [0 if value is None else value for value in counts]
    for value in counts:
        entropy += -value/sum(counts) * math.log(value/sum(counts), len(counts)) if value != 0 else 0
    return entropy
