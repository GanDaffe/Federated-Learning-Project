from federated_algo.base.base_client import BaseClient
from utils.training_utils import set_parameters, train

class FedAdmImpClient(BaseClient):
    def __init__(self, cid, cluster_id, net, criterion, trainloader, valloader):
        # print(f"[DEBUG] Type of net: {type(net)}")
        super().__init__(cid, net, criterion, trainloader, valloader)
        self.cluster_id = cluster_id
        
    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = train(self.net, self.trainloader, learning_rate=config["learning_rate"], loss_fn=self.criterion)
        return self.get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid, "cluster_id": self.cluster_id}
