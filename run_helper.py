import flwr as fl
import torch
from utils import get_parameters
from flwr.common import ndarrays_to_parameters
from clients import FlowerClient, FedAdmImpClient, FedProxClient, ScaffoldClient
from utils.training_utils import get_model
from import_fed import *

def run_simulation(
        algo: str, 
        model: str, 
        dataset: str, 
        trainloader, 
        valloader, 
        criterion,
        iids: int,
        num_round: int, 
        num_clients: int,  
        device,
        client_cluster_index = None,
        entropies = None    
    ):
    net = get_model(model, dataset).to(device)  

    def scaffold_client_fn(cid: str) -> ScaffoldClient:
        idx = int(cid)
        return ScaffoldClient(cid, net, trainloader=trainloader[idx], valloader=valloader[idx], criterion=criterion)
    
    def normal_client_fn(cid: str) -> FlowerClient:
        idx = int(cid)
        return FlowerClient(cid, net, trainloader=trainloader[idx], valloader=valloader[idx], criterion=criterion)
    
    def cluster_fed_client_fn(cid: str) -> FedAdmImpClient:
        idx = int(cid)
        return FedAdmImpClient(cid, net, cluster_id=client_cluster_index[idx], criterion=criterion, trainloader=trainloader[idx], valloader=valloader[idx])

    
    current_parameters = ndarrays_to_parameters(get_parameters(net))
    client_resources = {"num_cpus": 2, "num_gpus": 1} if device.type == "cuda" else {"num_cpus": 1, "num_gpus": 0.0}

    if algo == 'scaffold': 
        shapes = list(map(lambda arr: arr.shape, get_parameters(net)))

        fl.simulation.start_simulation(
            client_fn = scaffold_client_fn,
            num_clients = num_clients,
            config = fl.server.ServerConfig(num_rounds=num_round),
            strategy = SCAFFOLD(num_rounds=num_round,
                                num_clients=num_clients,
                                iids = iids,
                                current_parameters=current_parameters,
                                weight_shapes=shapes,
                                learning_rate = 0.1
                            ),

             client_resources = client_resources
        )  
    elif algo == 'fedadpimp': 
        assert entropies != None, f'Entropies for {algo} cannnot be none'
        fl.simulation.start_simulation(
            client_fn = cluster_fed_client_fn,
            num_clients = num_clients,
            config = fl.server.ServerConfig(num_rounds=num_round),
            strategy = BoxFedv2(num_rounds=num_round,
                                num_clients=num_clients,
                                iids = iids,
                                current_parameters=current_parameters,
                                entropies = entropies,
                                # learning_rate = 0.1
                            ),

            client_resources = client_resources
        )   
    elif algo == 'fedprox': 
        fl.simulation.start_simulation(
            client_fn = normal_client_fn-456,
            num_clients = num_clients,
            config = fl.server.ServerConfig(num_rounds=num_round),
            strategy = FedProx(num_rounds=num_round,
                               num_clients=num_clients,
                               iids = iids,
                               current_parameters=current_parameters,
                               # learning_rate = 0.1
                            ),
            client_resources = client_resources
        )
    elif algo == 'fedimp': 
        assert entropies != None, f'Entropies for {algo} cannnot be none'
        fl.simulation.start_simulation(
            client_fn = normal_client_fn,
            num_clients = num_clients,
            config = fl.server.ServerConfig(num_rounds=num_round),
            strategy = FedImp(num_rounds=num_round,
                              num_clients=num_clients,
                              iids = iids,
                              current_parameters=current_parameters,
                              entropies = entropies,
                              # learning_rate = 0.1
                            ),

            client_resources = client_resources
        )   
    elif algo == 'fedadp':
        fl.simulation.start_simulation(
            client_fn = normal_client_fn,
            num_clients = num_clients,
            config = fl.server.ServerConfig(num_rounds=num_round),
            strategy = FedImp(num_rounds=num_round,
                              num_clients=num_clients,
                              iids = iids,
                              current_parameters=current_parameters,
                              # learning_rate = 0.1
                            ),

            client_resources = client_resources
        )   
        
        