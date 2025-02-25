import flwr as fl
import torch
from torch import nn
from copy import deepcopy
from utils import get_parameters, set_parameters, train, test, train_scaffold
from collections import Counter, OrderedDict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, criterion, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = criterion

    def get_parameters(self, config):
        return get_parameters(self.net)
    
    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = train(self.net, self.trainloader, learning_rate=config["learning_rate"], loss_fn=self.criterion)
        return get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid, "cluster_id": self.cluster_id}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, loss_fn=self.criterion)
        return loss, len(self.valloader.sampler), {"accuracy": accuracy}



class FedAdmImpClient(fl.client.NumpyClient):
    def __init__(self, cid, cluster_id, net, criterion, trainloader, valloader):
        self.cid = cid
        self.cluster_id = cluster_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = criterion

    def get_parameters(self, config):
        return get_parameters(self.net)
        
    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = train(self.net, self.trainloader, learning_rate=config["learning_rate"], loss_fn=self.criterion)
        return get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid, "cluster_id": self.cluster_id}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, loss_fn=self.criterion)
        return loss, len(self.valloader.sampler), {"accuracy": accuracy}
    

class FedProxClient(fl.client.NumPyClient):
    def __init__(self, cid, net, criterion, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = criterion

    def get_parameters(self, config):
        return get_parameters(self.net)
    
    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = train(self.net, self.trainloader, learning_rate=config["learning_rate"], proximal_mu=config["proximal_mu"])
        return get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid, "cluster_id": self.cluster_id}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, loss_fn=self.criterion)
        return loss, len(self.valloader.sampler), {"accuracy": accuracy}


class ScaffoldClient(fl.client.NumPyClient):

    def __init__(self, cid, net, trainloader, valloader, criterion):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = criterion
        params = get_parameters(self.net)
        self.weight_shapes = list(map(lambda arr: arr.shape, params)) # used for checking weights and covariates

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):

        self.check_shapes(parameters, self.weight_shapes)

        third = len(parameters)//3
        server_weights = parameters[ : third]
        server_covariates = parameters[third : 2*third]
        client_covariates = parameters[2*third : ]

        set_parameters(self.net, server_weights)
        loss, accuracy, new_client_covariates = self._train(self.net, self.trainloader, config, server_covariates=server_covariates, client_covariates=client_covariates)
        weight_and_covariates = get_parameters(self.net) + new_client_covariates

        self.check_shapes(weight_and_covariates, self.weight_shapes)

        return weight_and_covariates, len(self.trainloader.dataset), {"loss": loss, "accuracy": accuracy, "id": self.cid}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, loss_fn=self.criterion)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}

    def _train(
        self,
        net,
        train_loader,
        config,
        server_covariates,
        client_covariates):

        learning_rate = config['learning_rate']
        weight_decay = float(config.get("weight_decay", 0.001))
        epochs = int(config.get("epochs", 1))
        optimizer = torch.optim.AdamW(
                net.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )

        original_server_weights = deepcopy(get_parameters(net))
        train_loss, train_acc = train_scaffold(
            net=net,
            trainloader=train_loader,
            loss_fn=self.criterion,
            optimizer=optimizer,
            epochs=epochs,
            server_covariates=server_covariates,
            client_covariates=client_covariates,
            learning_rate=learning_rate
        )

        #update client's covariates
        new_client_covariates = []
        weights = get_parameters(net)
        new_client_covariates = [client_covariates[i] - server_covariates[i] + (original_server_weights[i] - weights[i])/(epochs * learning_rate)
                                              for i in range(len(client_covariates))]

        return train_loss, train_acc, new_client_covariates
    

    def check_shapes(self, weights_and_covariates, weight_shapes) -> None:
        """Given a list of numpy arrays checks whether they have a repeating pattern of given shapes"""
        assert len(weights_and_covariates) % len(weight_shapes) == 0
        num_parts = len(weights_and_covariates) // len(weight_shapes)
        weights_or_covariates_parts = [
            weights_and_covariates[
                x * len(weight_shapes) : (x+1) * len(weight_shapes)
            ] for x in range(num_parts)
        ]
        for w_or_c_part in weights_or_covariates_parts:
            for i in range(len(w_or_c_part)):
                assert w_or_c_part[i].shape == weight_shapes[i]