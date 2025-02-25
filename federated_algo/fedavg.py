from __init__ import *

class FedAvg(fl.server.strategy.Strategy):
    def __init__(
        self,
        num_rounds: int,
        num_clients: int,
        iids,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        learning_rate: float = 0.1,
        decay_rate: float = 0.995,
        current_parameters: Optional[Parameters] = None
    ) -> None:
        super().__init__()
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.learning_rate = learning_rate
        self.iids = iids
        self.decay_rate = decay_rate
        self.current_parameters = current_parameters
        self.result = {"round": [], "train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": []}


    def __repr__(self) -> str:
        return "FedAvg"


    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.current_parameters


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        config = {"learning_rate": self.learning_rate}
        self.learning_rate *= self.decay_rate
        fit_ins = FitIns(parameters, config)

        fit_configs = [(client, fit_ins) for client in clients]
        return fit_configs


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
        self.current_parameters = ndarrays_to_parameters(aggregate(weights_results))
        metrics_aggregated = {}

        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        examples = [fit_res.num_examples for _, fit_res in results]
        loss = sum(losses) / sum(examples)
        accuracy = sum(corrects) / sum(examples)

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)
        print(f"train_loss: {loss} - train_acc: {accuracy}")

        return self.current_parameters, metrics_aggregated


    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
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
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""

        loss_aggregated = weighted_loss_avg([(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results])
        metrics_aggregated = {}

        corrects = [round(evaluate_res.num_examples * evaluate_res.metrics["accuracy"]) for _, evaluate_res in results]
        examples = [evaluate_res.num_examples for _, evaluate_res in results]
        accuracy = sum(corrects) / sum(examples)

        self.result["test_loss"].append(loss_aggregated)
        self.result["test_accuracy"].append(accuracy)
        print(f"test_loss: {loss_aggregated} - test_acc: {accuracy}")

        if server_round == self.num_rounds:
            df = pd.DataFrame(self.result)
            df.to_csv(f"result/fedavg{self.iids}.csv", index=False)
            
        return loss_aggregated, metrics_aggregated


    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters using an evaluation function."""
        return None


    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients


    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients



class FedAvgM(fl.server.strategy.Strategy):
    def __init__(
        self,
        num_rounds: int,
        num_clients: int,
        iids,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        learning_rate: float = 0.1,
        decay_rate: float = 0.995,
        server_learning_rate: float = 1.0,
        server_momentum: float = 0.0,
        current_parameters: Optional[Parameters] = None
    ) -> None:
        super().__init__()
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.learning_rate = learning_rate
        self.iids = iids
        self.decay_rate = decay_rate
        self.current_parameters = current_parameters
        self.server_momentum = server_momentum
        self.server_learning_rate = server_learning_rate
        self.server_opt: bool = (self.server_momentum != 0.0) or (
            self.server_learning_rate != 1.0
        )
        self.momentum_vector: Optional[NDArrays] = None
        self.result = {"round": [], "train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": []}

    def __repr__(self) -> str:
        return "FedAvgM"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return self.current_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        config = {"learning_rate": self.learning_rate}
        self.learning_rate *= self.decay_rate
        fit_ins = FitIns(parameters, config)

        fit_configs = [(client, fit_ins) for client in clients]
        return fit_configs

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        weights_results = [(parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
        fedavg_result = aggregate(weights_results)

        if self.server_opt:
            initial_weights = parameters_to_ndarrays(self.current_parameters)
            pseudo_gradient: NDArrays = [
                x - y
                for x, y in zip(
                    parameters_to_ndarrays(self.current_parameters), fedavg_result
                )
            ]
            if self.server_momentum > 0.0:
                if server_round > 1:
                    assert (
                        self.momentum_vector
                    ), "Momentum should have been created on round 1."
                    self.momentum_vector = [
                        self.server_momentum * x + y
                        for x, y in zip(self.momentum_vector, pseudo_gradient)
                    ]
                else:
                    self.momentum_vector = pseudo_gradient

                # No nesterov for now
                pseudo_gradient = self.momentum_vector

            # SGD
            fedavg_result = [
                x - self.server_learning_rate * y
                for x, y in zip(initial_weights, pseudo_gradient)
            ]
            # Update current weights
            self.initial_parameters = ndarrays_to_parameters(fedavg_result)

        self.current_parameters = ndarrays_to_parameters(fedavg_result)
        metrics_aggregated = {}

        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        examples = [fit_res.num_examples for _, fit_res in results]
        loss = sum(losses) / sum(examples)
        accuracy = sum(corrects) / sum(examples)

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)
        print(f"train_loss: {loss} - train_acc: {accuracy}")
        return self.current_parameters, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        evaluate_configs = [(client, evaluate_ins) for client in clients]
        return evaluate_configs

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        loss_aggregated = weighted_loss_avg([(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results])
        metrics_aggregated = {}

        corrects = [round(evaluate_res.num_examples * evaluate_res.metrics["accuracy"]) for _, evaluate_res in results]
        examples = [evaluate_res.num_examples for _, evaluate_res in results]
        accuracy = sum(corrects) / sum(examples)

        self.result["test_loss"].append(loss_aggregated)
        self.result["test_accuracy"].append(accuracy)
        print(f"test_loss: {loss_aggregated} - test_acc: {accuracy}")

        if server_round == self.num_rounds:
            df = pd.DataFrame(self.result)
            df.to_csv(f"result/fedavgm{self.iids}.csv", index=False)
            
        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients