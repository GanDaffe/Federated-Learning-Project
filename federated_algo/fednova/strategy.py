from __init__ import * 
from federated_algo.base.base_algo import FedAvg 

from logging import INFO, log

class FedNova(FedAvg):
    def __init__(self, exp_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exp_config = exp_config
        self.lr = self.exp_config.optimizer.lr 
        self.gmf = exp_config.optimizer.get("gmf", 0)  
        self.global_momentum_buffer = []  

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Tuple[ClientProxy, FitRes]],
    ) -> Tuple[Optional[Parameters], Dict[str, float]]:
        """Aggregate fit results using FedNova logic, kế thừa từ FedAvg."""

        local_tau = [fit_res.metrics["tau"] for _, fit_res in results]
        tau_eff = np.sum(local_tau)

        aggregate_parameters = []
        for _, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            scale = tau_eff / float(fit_res.metrics["local_norm"]) * float(fit_res.metrics["weight"])
            aggregate_parameters.append((params, scale))


        agg_cum_gradient = aggregate(aggregate_parameters)

        if self.current_parameters is None:
            self.current_parameters = ndarrays_to_parameters([np.zeros_like(g) for g in agg_cum_gradient])
        current_params_ndarrays = parameters_to_ndarrays(self.current_parameters)
        self.update_server_params(current_params_ndarrays, agg_cum_gradient)

        # Lưu kết quả train giống FedAvg
        losses = [fit_res.num_examples * fit_res.metrics["loss"] for _, fit_res in results]
        corrects = [round(fit_res.num_examples * fit_res.metrics["accuracy"]) for _, fit_res in results]
        examples = [fit_res.num_examples for _, fit_res in results]
        loss = sum(losses) / sum(examples) 
        accuracy = sum(corrects) / sum(examples) 

        self.result["round"].append(server_round)
        self.result["train_loss"].append(loss)
        self.result["train_accuracy"].append(accuracy)
        print(f"train_loss: {loss} - train_acc: {accuracy}")

        self.current_parameters = ndarrays_to_parameters(current_params_ndarrays)
        return self.current_parameters, {}

    def update_server_params(self, current_params: NDArrays, cum_grad: NDArrays):
        """Cập nhật tham số server dùng logic FedNova."""
        for i, (layer_param, layer_cum_grad) in enumerate(zip(current_params, cum_grad)):
            if self.gmf != 0:
                if len(self.global_momentum_buffer) <= i:
                    buf = layer_cum_grad / self.lr
                    self.global_momentum_buffer.append(buf)
                else:
                    self.global_momentum_buffer[i] = self.gmf * self.global_momentum_buffer[i] + layer_cum_grad / self.lr
                layer_param -= self.global_momentum_buffer[i] * self.lr
            else:
                layer_param -= layer_cum_grad

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
            df.to_csv(f"result/fednova{self.iids}.csv", index=False)
            
        return loss_aggregated, metrics_aggregated
