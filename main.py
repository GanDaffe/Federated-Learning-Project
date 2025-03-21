import flwr as fl
import torch
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from utils.preprocessing import load_data, partition_data, clustering
from utils.train_helper import compute_entropy
from run import run_simulation
from torch import nn
torch.cuda.manual_seed(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")


if __name__ == '__main__':
    disable_progress_bar()
    size_img = {
        'fmnist': 28, 
        'cifar10': 32, 
        'cifar100': 32,
        'emnist': 28,
        'sentimen140': None
    }
    
    input_size = {
        'fmnist': 1,
        'emnist': 1, 
        'cifar10': 3, 
        'cifar100': 3,
        'sentimen140': 1,
    }
    
    output_size = {
        'fmnist': 10, 
        'cifar10': 10, 
        'cifar100': 100, 
        'eminst': 37, 
        'sentimen140': 2, 
    }

    # ---------- HYPER PARAMETERS -------------
   

    experiment_config = {
        'algo':                     'fedadpimp',  #All letters in lowercase, no space
        'num_round':                500, 
        'num_clients':              50, 
        'iids':                     10, 
        'diff_non_iid':             20, # normal non-iid = num_clients - (iids + diff_non_iid)
        'batch_size':               100,
        'dataset_name':             'fmnist',  # emnist / fmnist / cifar10 / cifar100 / sentimen140 (take long time to load)
        'cluster_distance':         'hellinger', # hellinger / jensenshannon / cosine ... for fedadpimp experiment only
        'alpha':                    100, 
        'beta':                     0.01, 
        'device':                   DEVICE
    }

    model_config = {
        'model_name':               'cnn', 
        'out_shape':                output_size[experiment_config['dataset_name']], 
        'in_shape':                 input_size[experiment_config['dataset_name']],
        'hidden':                   32,
        'im_size':                  size_img[experiment_config['dataset_name']]
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
                                                min_smp=5,
                                                xi=0.2)

    num_cluster = len(list(set(client_cluster_index.values())))
    print(f'Number of Clusters: {num_cluster}')
    
    inc = 1
    for k, v in client_cluster_index.items():
        if v == -1:
            client_cluster_index[k] = num_cluster + inc
            inc += 1
            
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
    
    print(f'Testing on {experiment_config['algo']} and {model_config['model_name']}')
    run_simulation(
        experiment_config['algo'], 
        trainloaders,
        testloader, 
        client_cluster_index, 
        nn.CrossEntropyLoss(), 
        experiment_config, 
        entropies, 
        model_config,
        client_dataset_ratio
    )
    