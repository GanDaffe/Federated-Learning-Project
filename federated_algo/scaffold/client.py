from federated_algo.base.base_client import BaseClient 
from utils.training_utils import set_parameters
from __init__ import * 
from federated_algo.scaffold.scaffold_utils import set_c_local, load_c_local

class SCAFFOLD_CLIENT(BaseClient):
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
        return self.get_parameters(self.net), len(self.trainloader.dataset), results
    
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