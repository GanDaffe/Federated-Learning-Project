from algorithm.base.strategy import FedAvg 
from import_lib import * 
from algorithm.moon.moon_utils import test_moon

class MOON(FedAvg):

    def __init__(self, *args, temperature: float = 1.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def __repr__(self):
        return "MOON"

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        config = {"learning_rate": self.learning_rate, "temperature": self.temperature, "device": self.device}

        fit_ins = FitIns(parameters, config)

        fit_configs = [(client, fit_ins) for client in clients]
        return fit_configs

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""

        test_net = copy.deepcopy(self.net)
        set_parameters(test_net, parameters_to_ndarrays(parameters))
        
        loss, accuracy = test_moon(test_net, self.testloader, self.device)

        if server_round != 0:  
            self.result["test_loss"].append(loss)
            self.result["test_accuracy"].append(accuracy)
        
        print(f"test_loss: {loss} - test_acc: {accuracy}")

        if server_round == self.num_rounds:
            df = pd.DataFrame(self.result)
            df.to_csv(f"result/{self.algo_name}_{self.iids}.csv", index=False)

        return float(loss), {"accuracy": accuracy}