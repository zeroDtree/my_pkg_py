import hashlib
import json
import os


def save_hash_to_file(model, data=None, algorithm=None, filename="hash.txt"):
    def compute_hash(obj):
        return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

    model_hash = str(model)
    data_hash = str(data)
    algorithm_hash = str(algorithm)
    if model is not None:
        model_hash = compute_hash(model)
    if data is not None:
        data_hash = compute_hash(data)
    if algorithm is not None:
        algorithm_hash = compute_hash(algorithm)
    hash_dict = {
        "model_hash": model_hash,
        "data_hash": data_hash,
        "algorithm_hash": algorithm_hash,
    }
    # if not exist,then create it
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            pass
    # save hash to file
    with open(filename, "a") as f:
        json.dump(hash_dict, f, indent=4)
        f.write("\n")
