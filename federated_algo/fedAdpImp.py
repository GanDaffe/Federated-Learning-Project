from __init__ import * 

class BoxFedv2(fl.server.strategy.Strategy):
    def __init__(
        self,
        num_rounds: int,
        num_clients: int,
        entropies: List[float],
        iids,
        temperature: float = 1.5,
        alpha: int = 5,
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
        self.entropies = entropies
        self.temperature = temperature
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.temperature = temperature
        self.iids = iids
        self.current_parameters = current_parameters
        self.current_angles = [None] * num_clients
        self.alpha = alpha
        self.result = {"round": [], "train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": []}


    def  __repr__(self) -> str:
        return "BoxFed"


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

    def aggregate_cluster(self, cluster_clients: List[FitRes]):
        weight_results = [(parameters_to_ndarrays(fit_res.parameters),
                            fit_res.num_examples * np.exp(self.entropies[int(fit_res.metrics["id"])]/self.temperature))
                            for fit_res in cluster_clients]
        losses = [fit_res.num_examples * fit_res.metrics["loss"] for fit_res in cluster_clients]
        correct = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for fit_res in cluster_clients]
        examples = [fit_res.num_examples for fit_res in cluster_clients]
        loss = sum(losses) / sum(examples)
        accuracy = sum(correct) / sum(examples)

        aggregated_params = ndarrays_to_parameters(aggregate(weight_results))

        total_examples = sum(fit_res.num_examples for fit_res in cluster_clients)

        representative_metrics = dict(cluster_clients[0].metrics)
        representative_metrics["loss"] = loss
        representative_metrics["accuracy"] = accuracy

        # print([fit_res.metrics["id"] for fit_res in cluster_clients])
        return FitRes(parameters=aggregated_params,
                      num_examples=total_examples,
                      metrics=representative_metrics,
                      status=Status(code=0, message="Aggregated successfully")
                    )


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        cluster_data = {}

        for client_res in results:
            client, fit_res = client_res
            cluster_id = fit_res.metrics["cluster_id"]
            if cluster_id not in cluster_data:
                cluster_data[cluster_id] = []

            cluster_data[cluster_id].append(fit_res)

        cluster_results = {}

        for cluster_id, cluster_clients in cluster_data.items():
            if not cluster_id == 0:
                cluster_results[cluster_id] = self.aggregate_cluster(cluster_data[cluster_id])

        weights_results = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in cluster_results.items()]

        num_examples = [fit_res.num_examples for _, fit_res in cluster_results.items()]
        ids = [int(fit_res.metrics["id"]) for _, fit_res in cluster_results.items()]

        for fit_res in cluster_data[0]:
            weights_results.append(parameters_to_ndarrays(fit_res.parameters))
            num_examples.append(fit_res.num_examples)
            ids.append(int(fit_res.metrics["id"]))

        local_updates = np.array(weights_results, dtype=object) - np.array(parameters_to_ndarrays(self.current_parameters), dtype=object)

        local_gradients = -local_updates/self.learning_rate
        self.learning_rate *= self.decay_rate

        global_gradient = np.sum(np.array(num_examples).reshape(len(num_examples), 1) * local_gradients, axis=0) / sum(num_examples)

        local_grad_vectors = [np.concatenate([arr for arr in local_gradient], axis = None)
                              for local_gradient in local_gradients]

        global_grad_vector = np.concatenate([arr for arr in global_gradient], axis = None)

        instant_angles = np.arccos([np.dot(local_grad_vector, global_grad_vector) / (np.linalg.norm(local_grad_vector) * np.linalg.norm(global_grad_vector))
                          for local_grad_vector in local_grad_vectors])

        if server_round == 1:
            smoothed_angles = instant_angles
        else:
            pre_angles = [self.current_angles[i] for i in ids]
            smoothed_angles = [(server_round-1)/server_round * x + 1/server_round * y if x is not None else y
                               for x, y in zip(pre_angles, instant_angles)]

        for id, i in zip(ids, range(len(ids))):
            self.current_angles[id] = smoothed_angles[i]

        maps = self.alpha*(1-np.exp(-np.exp(-self.alpha*(np.array(smoothed_angles)-1))))

        weights = num_examples * np.exp(maps) / sum(num_examples * np.exp(maps))

        parameters_aggregated = np.sum(weights.reshape(len(weights), 1) * np.array(weights_results, dtype=object), axis=0)

        self.current_parameters = ndarrays_to_parameters(parameters_aggregated)
        metrics_aggregated = {}

        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in cluster_results.items()]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in cluster_results.items()]
        loss = sum(losses) / sum(num_examples)

        for fit_res in cluster_data[0]:
            losses.append(fit_res.num_examples * fit_res.metrics["loss"])
            corrects.append(round(fit_res.num_examples * fit_res.metrics["accuracy"]))
            loss = sum(losses) / sum(num_examples)

        accuracy = sum(corrects) / sum(num_examples)
        print(f"train_loss: {loss} - train_acc: {accuracy}")

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)

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
        print(f"test_loss: {loss_aggregated} - test_acc: {accuracy}")

        self.result["test_loss"].append(loss_aggregated)
        self.result["test_accuracy"].append(accuracy)

        os.makedirs("result", exist_ok=True)
        if server_round == self.num_rounds:
            df = pd.DataFrame(self.result)
            df.to_csv(f"result/boxfed_v2{self.iids}.csv", index=False)

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