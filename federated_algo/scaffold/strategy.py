from __init__ import *
from flwr.server.strategy import FedAvg

class SCAFFOLD(FedAvg):

    def __init__(
        self,
        global_model,
        num_rounds,
        num_clients,
        iids,
        learning_rate = 0.1,
        weight_decay = 0.001,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        initial_parameters: Optional[Parameters] = None
    ) -> None:
        super().__init__()

        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.iids = iids
        self.global_learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters


        self.c_global = [torch.zeros_like(param) for param in global_model.parameters()]
        self.current_weights = parameters_to_ndarrays(self.initial_parameters)
        self.num_clients = num_clients

        self.result = {"round": [], "test_loss": [], "test_accuracy": []}

    def __repr__(self) -> str:
        return 'SCAFFOLD'

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""      
        return self.initial_parameters
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""

        return None

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


    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""

        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        evaluate_configs = [(client, evaluate_ins) for client in clients]
        return evaluate_configs

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        corrects = [round(evaluate_res.num_examples * evaluate_res.metrics["accuracy"]) for _, evaluate_res in results]
        examples = [evaluate_res.num_examples for _, evaluate_res in results]
        accuracy = sum(corrects) / sum(examples)
        print(f"test_loss: {loss_aggregated} - test_acc: {accuracy}")

        self.result["round"].append(server_round)
        self.result["test_loss"].append(loss_aggregated)
        self.result["test_accuracy"].append(accuracy)

        if server_round == self.num_rounds:
            df = pd.DataFrame(self.result)
            df.to_csv(f"result/scaffold_{self.iids}.csv", index=False)

        metrics_aggregated = {}

        return loss_aggregated, metrics_aggregated