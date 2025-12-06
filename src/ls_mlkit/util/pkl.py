import os
import pickle


def load_pickle_file(pickle_path: str):
    try:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find pickle file {pickle_path}")
        return None
    except Exception as e:
        print(f"Error loading pickle file {pickle_path}: {e}")
        return None

    return data


def save_pickle_file(data, pickle_path: str):
    try:
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        with open(pickle_path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving pickle file {pickle_path}: {e}")
