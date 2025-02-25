import flwr as fl
import os
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Status,
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union, Optional
from functools import partial, reduce

def check_shapes(weights_and_covariates: NDArrays, weight_shapes: List[Tuple[int, ...]]) -> None:
    """Given a list of numpy arrays checks whether they have a repeating pattern of given shapes"""
    assert len(weights_and_covariates) % len(weight_shapes) == 0
    num_parts = len(weights_and_covariates) // len(weight_shapes)
    weights_or_covariates_parts = [
        weights_and_covariates[
            x * len(weight_shapes) : (x+1) * len(weight_shapes)
        ] for x in range(num_parts)
    ]
    for w_or_c_part in weights_or_covariates_parts:
      for i in range(len(w_or_c_part)):
        assert w_or_c_part[i].shape == weight_shapes[i]
        
def aggregate(results: list[tuple[NDArrays, float]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples) in results)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime