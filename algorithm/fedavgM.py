from import_lib import *
from algorithm.base.strategy import FedAvg

class FedAvgM(FedAvg): 

    def __init__(
            self,
            *args,
            server_learning_rate: float = 1.0,
            server_momentum: float = 0.0,
            **kwargs):
        
        super().__init__(*args, **kwargs) 

        self.server_momentum = server_momentum
        self.server_learning_rate = server_learning_rate
        self.server_opt: bool = (self.server_momentum != 0.0) or (
            self.server_learning_rate != 1.0
        )

        self.momentum_vector: Optional[NDArrays] = None

    def __repr__(self) -> str:
        return "FedAvgM"
    
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
    
