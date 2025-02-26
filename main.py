import flwr as fl
import torch
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler
from utils.data_processing import load_data, partition_data, clustering
from utils.training_utils import compute_entropy
from run_helper import run_simulation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

if __name__ == '__main__':
    

    # ---------- HYPER PARAMETERS -------------

    
    NUM_ROUNDS = 800
    NUM_CLIENTS = 30
    NUM_IIDS = 6
    DIFF_DISTRIBUTION_NON_IID_CLIENTS = 12     # 2 * DIFF_DISTRIBUTION_NON_IID_CLIENTS + NUM_IIDS <= NUM_CLIENTS
    BATCH_SIZE = 100
    LOSS_FN   = 'multiclass' # multiclass / binary
    DATASET   = 'cifar10'    # emnist / fmnist / cifar10 / cifar100 / sentiment140 (take long time to load)
    MODEL     = 'cnn'     # resnet101 / resnet50 / vgg16 / mlp / cnn / lstm
    DISTANCE  = 'hellinger' # hellinger / jensenshannon / cosine ... 
    FED = 'fedadpimp' # fedadp / scaffold / fedadpimp / fedprox / fedimp


    # ----------- LOADING THE DATA -------------


    trainset, testset = load_data(DATASET)
    ids, dist = partition_data(trainset, num_clients=NUM_CLIENTS, _iid=NUM_IIDS, non_iid_diff=DIFF_DISTRIBUTION_NON_IID_CLIENTS, alpha=100, beta=0.01, dataset_name=DATASET)

    client_cluster_index, distrib_ = clustering(trainset.classes, len(trainset), NUM_CLIENTS, dist, distance=DISTANCE, min_smp=2)

    print(f'Number of Clusters: {len(list(set(client_cluster_index.values())))}')
    for k, v in client_cluster_index.items():
        print(f'Client {k + 1}: Cluster: {v}')

    for i in range(NUM_CLIENTS):
        print(f"Client {i+1}: {dist[i]}")

    entropies = [compute_entropy(dist[i]) for i in range(NUM_CLIENTS)]
    trainloaders = []
    valloaders = []

    base = len(testset) // NUM_CLIENTS 
    extra = len(testset) % NUM_CLIENTS

    val_length = [base + (1 if i < extra else 0) for i in range(NUM_CLIENTS)]
    valsets = random_split(testset, val_length)

    for i in range(NUM_CLIENTS):
        trainloaders.append(DataLoader(trainset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(ids[i])))
        valloaders.append(DataLoader(valsets[i], batch_size=BATCH_SIZE))


    # ------------ RUN SIMULATION ---------------


    run_simulation( algo=FED,
                    model=MODEL,
                    dataset=DATASET,
                    trainloader=trainloaders, 
                    valloader=valloaders,
                    criterion=LOSS_FN, 
                    iids=NUM_IIDS,
                    num_clients=NUM_CLIENTS,
                    device=DEVICE,
                    num_round=NUM_ROUNDS,
                    client_cluster_index=client_cluster_index,
                    entropies=entropies
                )
