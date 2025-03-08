import flwr as fl
from flwr.common import ndarrays_to_parameters
from utils.training_utils import get_model, get_parameters, load_config
from import_algo import *

def run_simulation(
        algo, 
        trainloaders,
        testloader,
        client_cluster_index,
        criterion,
        exp_config,
        entropies, 
        client_config, 
        model_config,
        strategy_config,
        client_dataset_ratio
    ):
    
    device = exp_config['device'] 
    model_name, dataset = model_config['model_name'], exp_config['dataset_name'] 
    net = get_model(model_name, dataset, model_config)

    def base_client_fn(cid: str): 
        idx = int(cid)
        return BaseClient(cid, net, trainloaders[idx], criterion, client_config).to_client()
    def fednova_client_fn(cid: str):
        idx = int(cid)
        return FedNovaClient(idx, net, trainloaders[idx], criterion, client_config, ratio=client_dataset_ratio).to_client()
    
    def cluster_fed_client_fn(cid: str) -> ClusterFedClient:
        idx = int(cid)
        return ClusterFedClient(cid, net, trainloaders[idx], criterion, client_config, client_cluster_index=client_cluster_index[idx]).to_client()
    
    def fedprox_client_fn(cid: str): 
        idx = int(cid)
        return BaseClient(cid, net, trainloaders[idx], criterion, client_config).to_client()
    
    def scaffold_client_fn(cid: str): 
        idx = int(cid)
        return BaseClient(cid, net, trainloaders[idx], criterion, client_config, c_local=exp_config['c_local']).to_client()
    
    current_parameters = ndarrays_to_parameters(get_parameters(net))
    client_resources = {"num_cpus": 2, "num_gpus": 1} if device == "cuda" else {"num_cpus": 1, "num_gpus": 0.0}    

    if algo == 'fednova': 
        fl.simulation.start_simulation(
            client_fn = fednova_client_fn,
            num_clients = exp_config['num_clients'],
            config = fl.server.ServerConfig(num_rounds=exp_config['num_round']),
            strategy = FedNovaStrategy( net = net,
                                        exp_config=strategy_config,
                                        testloader=testloader,
                                        num_rounds=exp_config['num_round'],
                                        num_clients=exp_config['num_clients'],
                                        iids = exp_config['iids'],
                                        current_parameters=current_parameters,
                                ),

            client_resources = client_resources
        )
    elif algo == 'fedadpimp': 
        assert entropies != None, f'Entropies for {algo} cannnot be none'
        fl.simulation.start_simulation(
            client_fn = cluster_fed_client_fn,
            num_clients = exp_config['num_clients'],
            config = fl.server.ServerConfig(num_rounds=exp_config['num_round']),
            strategy = BoxFedv2(num_rounds=exp_config['num_round'],
                                num_clients=exp_config['num_clients'],
                                iids = exp_config['iids'],
                                current_parameters=current_parameters,
                                entropies = entropies,
                                # learning_rate = 0.1
                            ),

            client_resources = client_resources
        )   
    elif algo == 'fedprox': 
        fl.simulation.start_simulation(
            client_fn = fedprox_client_fn,
            num_clients = exp_config['num_clients'],
            config = fl.server.ServerConfig(num_rounds=exp_config['num_round']),
            strategy = FedProx(num_rounds=exp_config['num_round'],
                               num_clients=exp_config['num_clients'],
                               iids = exp_config['iids'],
                               current_parameters=current_parameters,
                               # learning_rate = 0.1
                            ),
            client_resources = client_resources
        )
    elif algo == 'fedimp': 
        assert entropies != None, f'Entropies for {algo} cannnot be none'
        fl.simulation.start_simulation(
            client_fn = base_client_fn,
            num_clients = exp_config['num_clients'],
            config = fl.server.ServerConfig(num_rounds=exp_config['num_round']),
            strategy = FedImp(num_rounds=exp_config['num_round'],
                              num_clients=exp_config['num_clients'],
                              iids = exp_config['iids'],
                              current_parameters=current_parameters,
                              entropies = entropies,
                              # learning_rate = 0.1
                            ),

            client_resources = client_resources
        )   
    elif algo == 'fedadp':
        fl.simulation.start_simulation(
            client_fn = base_client_fn,
            num_clients = exp_config['num_clients'],
            config = fl.server.ServerConfig(num_rounds=exp_config['num_round']),
            strategy = FedAdp(num_rounds=exp_config['num_round'],
                              num_clients=exp_config['num_clients'],
                              iids = exp_config['iids'],
                              current_parameters=current_parameters,
                              # entropies=entropies
                              # learning_rate = 0.1
                            ),

            client_resources = client_resources
        )   
    elif algo == 'fedavg': 
        fl.simulation.start_simulation(
            client_fn = base_client_fn, 
            num_clients = exp_config['num_clients'],
            config = fl.server.ServerConfig(num_rounds=exp_config['num_round']),
            strategy = FedAvg(num_rounds=exp_config['num_round'],
                              num_clients=exp_config['num_clients'],
                              iids = exp_config['iids'],
                              current_parameters=current_parameters,
                              learning_rate=0.1
                             ),
            client_resources=client_resources
        )
    elif algo == 'scaffold': 
        fl.simulation.start_simulation(
            client_fn = scaffold_client_fn,
            num_clients = exp_config['num_clients'],
            config = fl.server.ServerConfig(num_rounds=exp_config['num_round']),
            strategy = SCAFFOLD(global_model=net,
                                num_rounds=exp_config['num_round'],
                                num_clients=exp_config['num_clients'],
                                iids = exp_config['iids'],
                                initial_parameters=current_parameters,
                                learning_rate = 0.1
                            ),

             client_resources = client_resources
        )  
