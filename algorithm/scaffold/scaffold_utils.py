import os
import numpy as np
import torch
def load_c_local(partition_id: int):
    path = "c_local_folder/" + str(partition_id) +".txt"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            c_delta_bytes = f.read()

        array = np.frombuffer(c_delta_bytes, dtype=np.float64)
        return array
    else:
        return None

# Custom function to serialize to bytes and save c_local variable inside a file
def set_c_local(partition_id: int, c_local):
    path = "c_local_folder/" + str(partition_id) +".txt"

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    c_local_list = []
    for param in c_local:
        c_local_list += param.flatten().tolist()

    c_local_numpy = np.array(c_local_list, dtype=np.float64)
    c_local_bytes = c_local_numpy.tobytes()

    with open(path, 'wb') as f:
        f.write(c_local_bytes)

def test_scaffold(net, testloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
