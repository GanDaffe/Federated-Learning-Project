import os
import numpy as np

def load_c_local(partition_id: int):
    path = "/federated_algo/scaffold/training_process_files/c_local_folder/" + str(partition_id) +".txt"
    if os.path.exists(path):
        with open(path, 'rb') as f:
            c_delta_bytes = f.read()

        array = np.frombuffer(c_delta_bytes, dtype=np.float64)
        return array
    else:
        return None

# Custom function to serialize to bytes and save c_local variable inside a file
def set_c_local(partition_id: int, c_local):
    path = "/federated_algo/scaffold/training_process_files/c_local_folder/" + str(partition_id) +".txt"

    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    c_local_list = []
    for param in c_local:
        c_local_list += param.flatten().tolist()

    c_local_numpy = np.array(c_local_list, dtype=np.float64)
    c_local_bytes = c_local_numpy.tobytes()

    with open(path, 'wb') as f:
        f.write(c_local_bytes)
