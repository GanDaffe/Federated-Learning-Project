from algorithm.base.strategy import FedAvg 
from import_lib import * 

class BoxFedv2(FedAvg): 

    def __init__(self, 
                 *args, 
                 entropies: List[float], 
                 temperature: float = 1.5, 
                 alpha: int = 5, 
                 **kwargs
    ): 
        super().__init__(*args, **kwargs) 

        self.entropies = entropies
        self.temperature = temperature 
        self.alpha = alpha 
        self.current_angles = {}

    
    def __repr__(self): 
        return 'FedAdpImp'
    
    
    def aggregate_cluster(self, cluster_id, cluster_clients: List[FitRes]):
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
        representative_metrics["id"] = cluster_id
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

        for cluster_id, _ in cluster_data.items():
            if len(cluster_data[cluster_id]) > 1:
                cluster_results[cluster_id] = self.aggregate_cluster(cluster_id, cluster_data[cluster_id])

        weights_results = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in cluster_results.items()]

        num_examples = [fit_res.num_examples for _, fit_res in cluster_results.items()]
        ids = [int(fit_res.metrics["id"]) for _, fit_res in cluster_results.items()]

        for _, v in cluster_data.items():
            if len(v) == 1:
                fit_res = v[0]
                weights_results.append(parameters_to_ndarrays(fit_res.parameters))
                num_examples.append(fit_res.num_examples)
                ids.append(int(fit_res.metrics["cluster_id"]))

        local_updates = np.array(weights_results, dtype=object) - np.array(parameters_to_ndarrays(self.current_parameters), dtype=object)

        local_gradients = -local_updates/self.learning_rate

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
        
        for _, v in cluster_data.items():
            if len(v) == 1:
                fit_res = v[0]
                losses.append(fit_res.num_examples * fit_res.metrics["loss"])
                corrects.append(round(fit_res.num_examples * fit_res.metrics["accuracy"]))
                loss = sum(losses) / sum(num_examples)

        loss = sum(losses) / sum(num_examples)
        accuracy = sum(corrects) / sum(num_examples)
        print(f"train_loss: {loss} - train_acc: {accuracy}")

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)

        return self.current_parameters, metrics_aggregated
