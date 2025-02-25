from __init__ import * 

class SCAFFOLD(fl.server.strategy.Strategy):
    def __init__(
        self,
        num_rounds: int,
        num_clients: int,
        iids: int,
        learning_rate: float = 0.1,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        weight_shapes: List[Tuple[int, ...]] = None,
        current_parameters: Optional[Parameters] = None,
    ) -> None:
        super().__init__()
        self.initial_parameters = parameters_to_ndarrays(current_parameters)

        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.iids = iids
        self.learning_rate = learning_rate
        self.weight_shapes = list(map(lambda arr: arr.shape, self.initial_parameters)) # shapes used for checking weights and covariates
        self.covariates = [np.zeros(shape) for shape in self.weight_shapes] # server covariates

        # store client_covariates on server side because clients are generated on demand. Instead of each client holding its state
        # as the Scaffold paper describes, we store the client_covariates on server side. This of course can be changed.
        self.clients_covariates: Dict[str, NDArrays] = {} # cid -> client_covariates

        self.covariates_zero = [np.zeros(shape) for shape in self.weight_shapes] # initial client covariates
        self.result = {"round": [], "train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": []}
        self.evaluate_fn = None

    def __repr__(self) -> str:
        return "SCAFFOLD"

    def _unpack_parameters(self, parameters: Parameters) -> Tuple[NDArrays, NDArrays]:
        """Extract weights and covariates from parameters"""
        weights_and_covariates = parameters_to_ndarrays(parameters)
        # print_shapes(weights_and_covariates, 'weights_and_covariates')
        self._check_shapes(weights_and_covariates)
        weights = weights_and_covariates[:len(weights_and_covariates)//2]
        covariates = weights_and_covariates[len(weights_and_covariates)//2:]
        return weights, covariates

    def _pack_weights_and_covariates(
        self,
        weights: NDArrays,
        server_covariates: Optional[NDArrays] = None,
        client_covariates: Optional[NDArrays] = None,
      ) -> Parameters:
        """Convert weights and covariates to parameters"""
        weights_and_covariates = (
            weights
          + (server_covariates if server_covariates is not None else [])
          + (client_covariates if client_covariates is not None else [])
        )
        self._check_shapes(weights_and_covariates)
        return ndarrays_to_parameters(weights_and_covariates)

    def _check_shapes(self, weights_and_covariates: NDArrays) -> None:
        """check shapes of weights and covariates"""
        check_shapes(weights_and_covariates, self.weight_shapes)

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self._pack_weights_and_covariates(self.initial_parameters, self.covariates)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        # unpack parameters
        weights, server_covariates = self._unpack_parameters(parameters)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom params per client
        fit_configurations = []
        config = {"learning_rate": self.learning_rate,
                  "weight_decay": 0.001,
                  "num_worker": 2,
                  "epochs": 1
                  }
        print(f"Server config: {config}")  # Debug
        for client in clients:
            # we need to send client_covariates specific to each client
            client_covariates = self.clients_covariates.get(client.cid, self.covariates_zero)
            parameters = self._pack_weights_and_covariates(weights, server_covariates, client_covariates)
            fit_configurations.append((client, FitIns(parameters, config)))

        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        weights_results = []
        for client_proxy, fit_res in results:
            weights, client_covariates = self._unpack_parameters(fit_res.parameters)

            self.clients_covariates[client_proxy.cid] = client_covariates

            weight = 1
            weights_results.append((weights, weight))

        # equation (5)(i) - Scaffold paper
        # no need to use previous weights because the server learning rate is 1
        # and so mathematically the update depends only on the updated weights from clients
        weights_aggregated = aggregate(weights_results)
        covariates_list = list(self.clients_covariates.values()) 
        # equation (5)(ii) - Scaffold paper
        # similar trick as previously - no need to use previous server covariates
        # server_covariates = list(np.sum(list(self.clients_covariates.values()), axis=0) / self.num_clients)
        server_covariates = [
            np.sum([client_cov[i] for client_cov in covariates_list], axis=0) / self.num_clients
            for i in range(len(self.weight_shapes))
        ]
        parameters_aggregated = self._pack_weights_and_covariates(weights_aggregated, server_covariates)
        metrics_aggregated = {}

        num_examples = [fit_res.num_examples for _, fit_res in results]

        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        loss = sum(losses) / sum(num_examples)
        accuracy = sum(corrects) / sum(num_examples)
        print(f"train_loss: {loss} - train_acc: {accuracy}")

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)

        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # unpack parameters
        weights, _server_covariates = self._unpack_parameters(parameters)

        if self.fraction_evaluate == 0.0:
            return []

        # for local evaluation, we don't need server_covariates neither client_covariates

        configs = {}
        evaluate_ins = EvaluateIns(ndarrays_to_parameters(weights), configs)
        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
        corrects = [round(evaluate_res.num_examples * evaluate_res.metrics["accuracy"]) for _, evaluate_res in results]
        examples = [evaluate_res.num_examples for _, evaluate_res in results]
        accuracy = sum(corrects) / sum(examples)
        print(f"test_loss: {loss_aggregated} - test_acc: {accuracy}")

        self.result["test_loss"].append(loss_aggregated)
        self.result["test_accuracy"].append(accuracy)

        os.makedirs("result", exist_ok=True)
        if server_round == self.num_rounds:
            df = pd.DataFrame(self.result)
            df.to_csv(f"result/scaffold_numiid_{self.iids}.csv", index=False)

        return loss_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None

        weights, server_covariates = self._unpack_parameters(parameters)
        eval_res = self.evaluate_fn(server_round, weights, {}) # using weights only for the evaluation
        if eval_res is None:
            return None

        return eval_res

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients