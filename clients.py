import flwr as fl
import torch
from torch import nn
from copy import deepcopy
from utils.training_utils import get_parameters, set_parameters, train, test, train_scaffold, load_c_local, set_c_local
from collections import Counter, OrderedDict
from flwr.client.numpy_client import NumPyClient  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(NumPyClient):
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
        return get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, loss_fn=self.criterion)
        return loss, len(self.valloader.sampler), {"accuracy": accuracy}



class FedAdmImpClient(FlowerClient):
    def __init__(self, cid, cluster_id, net, criterion, trainloader, valloader):
        # print(f"[DEBUG] Type of net: {type(net)}")
        super().__init__(cid, net, criterion, trainloader, valloader)
        self.cluster_id = cluster_id
        
    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = train(self.net, self.trainloader, learning_rate=config["learning_rate"], loss_fn=self.criterion)
        return get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid, "cluster_id": self.cluster_id}


class FedProxClient(FlowerClient):
    def __init__(self, cid, net, criterion, trainloader, valloader):
        super().__init__(cid, net, criterion, trainloader, valloader)
    
    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = train(self.net, self.trainloader, learning_rate=config["learning_rate"], proximal_mu=config["proximal_mu"])
        return get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid, "cluster_id": self.cluster_id}


class ScaffoldClient(FlowerClient):

    def __init__(self, cid, net, trainloader, valloader, criterion):
        super().__init__(cid, net, criterion, trainloader, valloader)

        params = get_parameters(self.net)
        self.weight_shapes = list(map(lambda arr: arr.shape, params)) # used for checking weights and covariates

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
                
class SCAFFOLD_CLIENT(NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, c_local, device):
        self.cid = cid
        self.net = net
        self.device=device
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.1, momentum=0.9)
        self.c_local = c_local

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_parameters(self.net, parameters)
        results = self.train_scaffold(
            net=self.net,
            trainloader=self.trainloader,
            valloader=self.valloader,
            epochs=1,
            learning_rate=0.1,
            device=self.device,
            config=config,
            c_local=self.c_local,
            parameters=parameters
        )
        return get_parameters(self.net), len(self.trainloader.dataset), results
    
    def train_scaffold(self, net, trainloader, valloader, epochs, learning_rate, device, config, c_local, parameters):
        """Train the model on the training set."""
        c_global_bytes = config['c_global']
        # Deserialize c_global list from bytes to float64
        c_global = np.frombuffer(c_global_bytes, dtype=np.float64)
        # Cache trainable global parameters
        global_weight = [param.detach().clone() for param in self.net.parameters()]
        if c_local is None:
            log(INFO, f"No cache found for c_local")
            c_local = [torch.zeros_like(param) for param in self.net.parameters()]

        net.to(device)  # Move model to GPU if available TODO: Make everything work on GPU.
        net.train()

        for _ in range(epochs):
            prebatch_params = [param.detach().clone() for param in self.net.parameters()]
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)

                self.optimizer.zero_grad()
                self.criterion(net(images.to(device)), labels.to(device)).backward()
                

                self.optimizer.step()

            # Local updates to the client model cf. Scaffold equation (n°3)
            # Adds Scaffold computation of c_diff in parameters
            for param, y_i, c_l, c_g in zip(self.net.parameters(), prebatch_params, c_local, c_global):
                if param.requires_grad:
                    param.grad.data = y_i - (learning_rate * (param.grad.data - c_l + c_g))

        # Update local control variate
        # Declare Scaffold variables
        y_delta = []
        c_plus = []
        c_delta = []

        # Local updates to the client control variate cf. Scaffold equation (n°4)
        # Compute c_plus : Option 2
        coef = 1 / (epochs * learning_rate)
        for c_l, c_g, param_l, param_g in zip(c_local, c_global, self.net.parameters(), global_weight):
            c_plus.append(c_l - c_g + ((param_g - param_l)*coef))

        # Compute c_plus : Option 1
        # c_plus_net = Net()
        # set_weights(c_plus_net, parameters)

        # criterion = torch.nn.CrossEntropyLoss().to(self.device)
        # optimizer = torch.optim.SGD(c_plus_net.parameters(), lr=learning_rate, momentum=0.9)

        # c_plus_net.to(device)  # move model to GPU if available
        # c_plus_net.train()
        # for batch in trainloader:
        #     images = batch["image"]
        #     labels = batch["label"]
        #     optimizer.zero_grad()
        #     criterion(c_plus_net(images.to(device)), labels.to(device)).backward()
        #     optimizer.step()

        # for x_param in c_plus_net.parameters():
        #     c_plus.append(x_param)

        # Compute y_delta (difference of model before and after training)
        for param_l, param_g in zip(self.net.parameters(), global_weight):
            y_delta.append(param_l - param_g)

        # Erase net params with y_delta params for weight averaging in FedAvg
        for param, new_w in zip(self.net.parameters(), y_delta):
            param.data = new_w.clone().detach() 

        # Compute c_delta
        for c_p, c_l in zip(c_plus, c_local):
            c_delta.append(c_p - c_l)

        set_c_local(self.cid, c_plus)

        # Create a bytes stream for c_delta
        # Flatten list to be compatible with numpy
        c_delta_list = []
        for param in c_delta:
            c_delta_list += param.flatten().tolist()
            
        c_delta_numpy = np.array(c_delta_list, dtype=np.float64)
        # Serialize to bytes
        c_delta_bytes = c_delta_numpy.tobytes()

        val_loss, val_acc = self.test_scaffold(net, valloader, device)

        results = { 
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "c_delta": c_delta_bytes,
        }
        return results

    def test_scaffold(self, net, testloader, device):
        """Validate the model on the test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        loss = loss / len(testloader)
        return loss, accuracy
    
    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_parameters(self.net, parameters)
        loss, accuracy = self.test_scaffold(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}
