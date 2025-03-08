import flwr as fl
import os
import copy
import torch
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
from collections import OrderedDict
from typing import List, Tuple, Dict, Union, Optional
from functools import partial, reduce
from utils.train_helper import train, get_parameters, set_parameters

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
        