from import_lib import *

class BaseClient(fl.client.NumPyClient):
    def __init__(self, 
                 cid, 
                 net, 
                 trainloader, 
                 criterion): 
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.client_config = client_config
        self.criterion = criterion

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        optimizer = torch.optim.SGD(params=self.net.parameters(), lr=config['learning_rate'])
        loss, accuracy = train(self.net, self.trainloader, self.criterion, optimizer, device=config['device'])
        return self.get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid}
    
    def evaluate(self, parameters, config):
        return None
