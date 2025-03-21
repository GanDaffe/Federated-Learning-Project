from import_lib import * 
from algorithm.base.client import BaseClient 

class ClusterFedClient(BaseClient): 

    def __init__(self, *args, cluster_id, **kwargs):
        super().__init__(*args, **kwargs)

        self.cluster_id = cluster_id 

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        optimizer = torch.optim.SGD(params=self.net.parameters(), lr=config['learning_rate'])
        loss, accuracy = train(self.net, self.trainloader, self.criterion, optimizer, device=config['device'])
        
        return self.get_parameters(self.net), len(self.trainloader.sampler), {"loss": loss, "accuracy": accuracy, "id": self.cid, "cluster_id": self.cluster_id}
