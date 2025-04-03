import functools
import hashlib
import os
import pickle
from functools import wraps
from datetime import datetime
import wandb


# decorator（arg）（func）


def cache_to_disk(root_datadir="cached_dataset"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not os.path.exists(root_datadir):
                os.makedirs(root_datadir)

            func_name = func.__name__.replace("/", "")
            cache_filename = root_datadir + "/" + f"{func_name}.pkl"
            args_str = "_".join(map(str, args))
            kwargs_str = "_".join(f"{k}={v}" for k, v in kwargs.items())
            params_str = f"{args_str}_{kwargs_str}"
            params_hash = hashlib.md5(params_str.encode()).hexdigest()
            cache_filename = os.path.join(root_datadir, f"{func_name}_{params_hash}.pkl")
            print("cache_filename =", cache_filename)

            if os.path.exists(cache_filename):
                with open(cache_filename, "rb") as f:
                    print(f"Loading cached data for {func.__name__} {params_str}")
                    return pickle.load(f)

            result = func(*args, **kwargs)

            print("caching " + cache_filename)
            with open(cache_filename, "wb") as f:
                pickle.dump(result, f)
                print(f"Cached data for {func.__name__}")

            hash_table_filename = os.path.join(root_datadir, "hash_table.txt")
            if not os.path.exists(hash_table_filename):
                with open(hash_table_filename, "w"):
                    pass
            with open(hash_table_filename, "a") as f:
                f.write(f"{cache_filename}: {params_str}\n")

            return result

        return wrapper

    return decorator


def timer(data_format="ms"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            begin_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            cost = (end_time - begin_time).seconds
            print(
                func.__name__ + " run" + f" {cost // 60} min {cost % 60}s",
            )
            return result

        return wrapper

    return decorator


def wandb_logger():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            wandb.log(result)
            return result

        return wrapper

    return decorator


def register_class_to_dict(cls=None, *, key_name=None, global_dict=None):
    """A decorator for registering classes to a global dictionary."""

    def _register(cls):
        if key_name is None:
            local_key_name = cls.__name__
        else:
            local_key_name = key_name
        if key_name in global_dict:
            raise ValueError(f"Already registered model with name: {key_name}")
        global_dict[local_key_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


if __name__ == "__main__":
    pass
