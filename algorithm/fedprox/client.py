from algorithm.base.client import BaseClient
from utils.train_helper import set_parameters, train

class FedProxClient(BaseClient):
    def __init__(self, cid, net, criterion, trainloader, valloader):
        super().__init__(cid, net, criterion, trainloader, valloader)
    
    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = train(self.net, self.trainloader, learning_rate=config["learning_rate"], proximal_mu=config["proximal_mu"])
        return self.get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid}