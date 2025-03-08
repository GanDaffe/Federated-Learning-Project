from import_lib import *
from algorithm.base.strategy import FedAvg
from algorithm.scaffold.scaffold_utils import test_scaffold
class SCAFFOLD(FedAvg):

    def __init__(
        self,
        *args,
        weight_decay=0.1, 
        **kwargs, 
    ) -> None:
        super().__init__(*args, **kwargs)

       
        self.weight_decay = weight_decay

        self.c_global = [torch.zeros_like(param) for param in self.net.parameters()]
        self.current_weights = parameters_to_ndarrays(self.current_parameters)
        self.num_clients = self.num_clients
        self.global_learning_rate = self.learning_rate
        self.result = {"round": [], "train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": []}

    def __repr__(self) -> str:
        return 'SCAFFOLD'


    def configure_fit(
        self, server_round, parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # Get the standard client/config pairs from the FedAvg super-class
        client_config_pairs = super().configure_fit(
              server_round, parameters, client_manager
          )
        # Serialize c_global to be compatible with config FitIns return values
        c_global = []
        for param in self.c_global:
            c_global += param.flatten().tolist()  # Flatten all params
            
        global_c_numpy = np.array(c_global, dtype=np.float64)
        global_c_bytes = global_c_numpy.tobytes()

        # Return client/config pairs with the c_global serialized control variate
        return [
            (
                client,
                FitIns(
                    fit_ins.parameters,
                    {**fit_ins.config, "c_global": global_c_bytes},
                ),
            )
            for client, fit_ins in client_config_pairs
        ]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        if parameters_aggregated is None:
            return None, {}

        fedavg_weights_aggregate = parameters_to_ndarrays(parameters_aggregated)

        # Aggregating the updates of y_delta to current weight cf. Scaffold equation (n°5)
        for current_weight, fed_weight in zip(self.current_weights, fedavg_weights_aggregate):
            current_weight += fed_weight * self.global_learning_rate

        # Initalize c_delta_sum for the weight average
        c_delta_sum = [np.zeros_like(c_global) for c_global in self.c_global]

        for _, fit_res in results:
            # Getting serialized buffer from fit metrics 
            c_delta = np.frombuffer(fit_res.metrics["c_delta"], dtype=np.float64)
            # Sum all c_delta in a single weight vector
            for i in range(len(c_delta_sum)):
                c_delta_sum[i] += np.array(c_delta[i], dtype=np.float64)

        for i in range(len(self.c_global)):
            # Aggregating the updates of c_global cf. Scaffold equation (n°5)
            c_delta_avg = c_delta_sum[i] / self.num_clients
            self.c_global[i] += torch.tensor(c_delta_avg)

        return ndarrays_to_parameters(self.current_weights), metrics_aggregated


    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        metrics_aggregated = {}

        loss, metrics = self.evaluate(server_round, self.current_parameters)

        self.result["test_loss"].append(loss)
        self.result["test_accuracy"].append(metrics['accuracy'])
        print(f"test_loss: {loss} - test_acc: {metrics['accuracy']}")

        if server_round == self.num_rounds:
            df = pd.DataFrame(self.result)
            df.to_csv(f"result/scaffold_{self.iids}.csv", index=False)

        return loss, metrics_aggregated
    
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""

        test_net = copy.deepcopy(self.net)  
        set_parameters(test_net, parameters_to_ndarrays(parameters))    
       
        loss, accuracy = test_scaffold(test_net, self.testloader)

        return float(loss), {"accuracy": accuracy}
