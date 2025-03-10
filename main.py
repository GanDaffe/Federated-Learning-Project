import flwr as fl
import torch
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from utils.preprocessing import load_data, partition_data, clustering
from utils.train_helper import compute_entropy
from run import run_simulation
from torch import nn
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

if __name__ == '__main__':
    

    # ---------- HYPER PARAMETERS -------------
    size_img = {
        'fmnist': 28, 
        'cifar10': 32, 
        'cifar100': 32,
        'emnist': 28
    }
    
    experiment_config = {
        'algo':                     'fednova',  #All letters in lowercase, no space
        'num_round':                3, 
        'num_clients':              30, 
        'iids':                     6, 
        'diff_non_iid':             12, 
        'batch_size':               100,
        'dataset_name':             'fmnist',  # emnist / fmnist / cifar10 / cifar100 / sentimen140 (take long time to load)
        'cluster_distance':         'hellinger', # hellinger / jensenshannon / cosine ... for fedadpimp experiment only
        'alpha':                    100, 
        'beta':                     0.01, 
        'device':                   DEVICE
    }

    model_config = {
        'model_name':               'cnn', 
        'num_layer':                2,  # For CNN only
        'out_shape':                10, # 2 for sentimen140, 100 for cifar100 otherwise 10
        'in_shape':                 1,
        'hidden':                   32,
        'im_size':                  size_img[experiment_config['dataset_name']]
    }
    
    client_config =  {
        'device':                   DEVICE, 
        "var_local_epochs":         True,  # Whether to use variable local epochs
        "var_min_epochs":           1,  # Minimum number of local epochs
        "var_max_epochs":           5,  # Maximum number of local epochs
        "seed":                     42,  # Random seed for reproducibility
        "optimizer": {
            "lr":       0.01,  # Learning rate
            "momentum": 0.9,  # Momentum for SGD
            "mu":       0.005,  # Proximal term (adjusted in FedNovaClient if needed)
        }
    }
    

    strategy_config = {
        'gmf':                      0, 
        'optimizer': {
            'gmf':      0              
        },
        'device':                   DEVICE
    }


    # ----------- LOADING THE DATA -------------


    trainset, testset = load_data(experiment_config['dataset_name'])
    ids, dist = partition_data(trainset, 
                               num_clients=experiment_config['num_clients'], 
                               _iid=experiment_config['iids'], 
                               non_iid_diff=experiment_config['diff_non_iid'], 
                               alpha=experiment_config['alpha'], 
                               beta=experiment_config['beta'], 
                               dataset_name=experiment_config['dataset_name'])

    client_cluster_index, distrib_ = clustering(trainset.classes, 
                                                len(trainset), 
                                                experiment_config['num_clients'], 
                                                dist, 
                                                distance=experiment_config['cluster_distance'], 
                                                min_smp=2)

    print(f'Number of Clusters: {len(list(set(client_cluster_index.values())))}')
    for k, v in client_cluster_index.items():
        print(f'Client {k + 1}: Cluster: {v}')

    for i in range(experiment_config['num_clients']):
        print(f"Client {i+1}: {dist[i]}")

    entropies = [compute_entropy(dist[i]) for i in range(experiment_config['num_clients'])]
    trainloaders = []
    testloader = DataLoader(testset, batch_size=experiment_config['batch_size'])

    for i in range(experiment_config['num_clients']):
        trainloaders.append(DataLoader(trainset, batch_size=experiment_config['batch_size'], sampler=SubsetRandomSampler(ids[i])))
       

    client_dataset_ratio: float = int((len(trainset) / experiment_config['num_clients'])) / len(trainset)

    # ------------ RUN SIMULATION ---------------

    run_simulation(
        experiment_config['algo'], 
        trainloaders,
        testloader, 
        client_cluster_index, 
        nn.CrossEntropyLoss(), 
        experiment_config, 
        entropies, 
        client_config, 
        model_config,
        strategy_config,
        client_dataset_ratio
    )
