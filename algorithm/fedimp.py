from import_lib import * 
from algorithm.base.strategy import FedAvg

class FedImp(FedAvg):
    def __init__(
        self,
        *args,
        entropies: List[float],
        temperature: float = 1.5,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.entropies = entropies
        self.temperature = temperature
    
        self.result = {"round": [], "train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": []}


    def __repr__(self) -> str:
        return "FedImp"


    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

        weights_results = [(parameters_to_ndarrays(fit_res.parameters),
                            fit_res.num_examples * np.exp(self.entropies[int(fit_res.metrics["id"])]/self.temperature))
                            for _, fit_res in results]
        print([fit_res.metrics["id"] for _, fit_res in results])
        self.current_parameters = ndarrays_to_parameters(aggregate(weights_results))
        metrics_aggregated = {}

        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        examples = [fit_res.num_examples for _, fit_res in results]
        loss = sum(losses) / sum(examples)
        accuracy = sum(corrects) / sum(examples)
        print(f"train_loss: {loss} - train_acc: {accuracy}")

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)

        return self.current_parameters, metrics_aggregated
