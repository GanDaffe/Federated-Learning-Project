import flwr as fl
import torch
from flwr.common import ndarrays_to_parameters
from utils.training_utils import get_model, get_parameters, load_config
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
        client_dataset_ratio,
        client_cluster_index = None,
        entropies = None    
    ):
    net = get_model(model, dataset).to(device)  
    # print(f"Type of net: {type(net)}")
    # print(f"Expected model but got: {net}")

    def scaffold_client_fn(cid: str) -> SCAFFOLD_CLIENT:
        idx = int(cid)
        return SCAFFOLD_CLIENT(cid, net, trainloader[idx], valloader[idx], criterion, device).to_client()
    
    def normal_client_fn(cid: str):
        idx = int(cid)
        return BaseClient(cid, net,  criterion, trainloader[idx], valloader[idx]).to_client()
    
    def cluster_fed_client_fn(cid: str) -> FedAdmImpClient:
        idx = int(cid)
        return FedAdmImpClient(cid, client_cluster_index[idx], net, criterion, trainloader[idx], valloader[idx]).to_client()

    def fednova_client_fn(cid: str): 
        idx = int(cid)
        return FedNovaClient(cid, net, trainloader, valloader, loss_type=criterion, device=device, config=load_config(), ratio=client_dataset_ratio)

    def fedprox_client_fn(cid: str): 
        idx = int(cid)
        return FedProxClient(cid, net,  criterion, trainloader[idx], valloader[idx]).to_client()
    
    current_parameters = ndarrays_to_parameters(get_parameters(net))
    client_resources = {"num_cpus": 2, "num_gpus": 1} if device.type == "cuda" else {"num_cpus": 1, "num_gpus": 0.0}

    if algo == 'scaffold': 

        fl.simulation.start_simulation(
            client_fn = scaffold_client_fn,
            num_clients = num_clients,
            config = fl.server.ServerConfig(num_rounds=num_round),
            strategy = SCAFFOLD(global_model=net,
                                num_rounds=num_round,
                                num_clients=num_clients,
                                iids = iids,
                                initial_parameters=current_parameters,
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
            client_fn = fedprox_client_fn,
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
            strategy = FedAdp(num_rounds=num_round,
                              num_clients=num_clients,
                              iids = iids,
                              current_parameters=current_parameters,
                              # entropies=entropies
                              # learning_rate = 0.1
                            ),

            client_resources = client_resources
        )   
    elif algo == 'fednova': 
        fl.simulation.start_simulation(
            client_fn = fednova_client_fn,
            num_clients = num_clients,
            config = fl.server.ServerConfig(num_rounds=num_round),
            strategy = FedNova( exp_config=load_config(),
                                num_rounds=num_round,
                                num_clients=num_clients,
                                iids = iids,
                                current_parameters=current_parameters,
                            ),

            client_resources = client_resources
        )
    elif algo == 'fedavg': 
        fl.simulation.start_simulation(
            client_fn = normal_client_fn, 
            num_clients = num_clients, 
            config = fl.server.ServerConfig(num_round=num_round), 
            strategy= FedAvg( num_rounds=num_round, 
                             num_clients=num_clients, 
                             iids=iids,
                             current_parameters=current_parameters,
                             learning_rate=0.1
                             ),
        )
