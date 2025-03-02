from utils.training_utils import get_parameters, set_parameters, train, test
import flwr as fl

class BaseClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, loss_type, device): 
        self.cid = cid
        self.net = net
        self.trainloader = trainloader 
        self.valloader = valloader
        
        self.loss_type = loss_type
        self.device = device


    def get_parameters(self, config): 
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = train(self.net, self.trainloader, learning_rate=config["learning_rate"], loss_fn=LOSS_FN)
        #loss, accuracy = utils.train(self.net, self.trainloader, learning_rate=config["learning_rate"], proximal_mu=config["proximal_mu"])
        return get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid, "cluster_id": self.cluster_id}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, loss_fn=self.loss_type)
        return loss, len(self.valloader.sampler), {"accuracy": accuracy}
