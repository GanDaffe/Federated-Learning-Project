from federated_algo.base.base_client import BaseClient
from __init__ import * 
from fednova_utils import ProxSGD, train_fednova, test_fednova

class FedNovaClient(BaseClient): 
    def __init__(self, *args, config, ratio, **kwargs):  
        super().__init__(*args, **kwargs) 
        self.exp_config = config
        self.ratio = ratio
        self.optimizer = ProxSGD(params=self.net.parameters(),
                                 lr=self.exp_config.optimizer.lr,
                                 momentum=self.exp_config.optimizer.momentum,
                                 mu=self.exp_config.optimizer.mu,

                                 ratio=self.ratio)
        
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        params = [
            val["cum_grad"].cpu().numpy()
            for _, val in self.optimizer.state_dict()["state"].items()
        ]
        return params

    def set_parameters(self, parameters): 
        self.optimizer.set_model_params(parameters)

    def fit(self, parameters, config):

        self.set_parameters(parameters) 
        learning_rate = self.exp_config.optimizer.lr 
        num_epochs = self.exp_config.var_min_epochs

        train_loss, train_acc = train_fednova(
            self.net,
            self.optimizer,
            self.trainloader,
            self.device,
            num_epochs,
            proximal_mu=self.exp_config.optimizer.mu,
        )

        grad_scaling_factor = self.optimizer.get_gradient_scaling()

        metrics = {
            "accuracy": train_acc,
            "loss": train_loss,
            "tau": grad_scaling_factor["tau"],      # Extract scalar values
            "local_norm": grad_scaling_factor["local_norm"],
            "weight": grad_scaling_factor["weight"]
        }

        return self.get_parameters({}), len(self.trainloader.sampler), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, metrics = test_fednova(self.net, self.valloader, self.device)
        # print(loss)
        return float(loss), len(self.valloader.sampler), metrics

    