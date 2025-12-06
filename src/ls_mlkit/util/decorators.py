import functools
import hashlib
import os
import pickle
from datetime import datetime
from functools import wraps
from typing import Callable, Dict, List

# decorator（arg）（func）


def cache_to_disk(root_datadir="cached_dataset", exclude_first_arg=False):
    """Cache the result of a function to disk

    Args:
        root_datadir (str, optional): the root directory to save the cached data. Defaults to "cached_dataset".
        exclude_first_arg (bool, optional): whether to exclude the first argument of the function when generating the cache filename. Defaults to False.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not os.path.exists(root_datadir):
                os.makedirs(root_datadir)

            func_name = func.__name__.replace("/", "")
            cache_filename = root_datadir + "/" + f"{func_name}.pkl"
            args_str = "_".join(map(str, args[1:] if exclude_first_arg else args))
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


def timer(format="ms"):
    """Timer the execution time of a function

    Args:
        format (str, optional): the format of the execution time. Defaults to "ms".
    """

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


def register_class_to_dict(cls=None, *, key_name=None, global_dict=None):
    """Register a class to a global dictionary

    Args:
        cls (class, optional): the class to register. Defaults to None.
        key_name (str, optional): the name of the key to register the class. Defaults to None.
        global_dict (dict, optional): the global dictionary to register the class. Defaults to None.

    """

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


def class_decorator(
    decorate_dict: Dict[str, List[Callable]] = {},
):
    def _class_decorator(cls):
        for attr_name, decorators in decorate_dict.items():
            for decorator in decorators:
                if attr_name in cls.__dict__:
                    method = getattr(cls, attr_name)
                    if callable(method):
                        print(attr_name, type(decorator))
                        setattr(cls, attr_name, decorator(method))

        return cls

    return _class_decorator


def require_keys(*required_keys):
    """Decorator to ensure returned dictionary contains required keys"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not isinstance(result, dict):
                raise TypeError(f"{func.__name__} must return a dictionary")

            missing_keys = set(required_keys) - result.keys()
            if missing_keys:
                raise KeyError(f"{func.__name__} missing required keys: {missing_keys}")

            return result

        return wrapper

    return decorator


def inherit_docstrings(cls):
    r"""
    Class decorator that automatically inherits docstrings from parent class methods.

    This decorator will:
    1. Find methods in the class that don't have docstrings
    2. Look for the same method in parent classes
    3. Copy the docstring from the first parent class that has one
    4. Handle methods marked with @inherit_docstring_from_parent

    Usage:

        .. code-block:: python

            @inherit_docstrings
            class ChildClass(ParentClass):
                def some_method(self):
                    # This method will inherit docstring from ParentClass.some_method
                    pass

                @inherit_docstring_from_parent('parent_method')
                def child_method(self):
                    # This method will inherit docstring from ParentClass.parent_method
                    pass

    Args:
        cls: The class to apply docstring inheritance to

    Returns:
        The modified class with inherited docstrings
    """
    for attr_name in dir(cls):
        # Skip private/magic methods and non-callable attributes
        if attr_name.startswith("_"):
            continue

        attr = getattr(cls, attr_name)
        if not callable(attr):
            continue

        # Check if this method exists in the class's own __dict__ (not inherited)
        if attr_name not in cls.__dict__:
            continue

        # Handle methods marked with @inherit_docstring_from_parent
        if hasattr(attr, "_inherit_docstring_from"):
            target_method_name = attr._inherit_docstring_from
            # Look for the target method in parent classes
            for base in cls.__mro__[1:]:  # Skip the class itself
                if hasattr(base, target_method_name):
                    parent_method = getattr(base, target_method_name)
                    if callable(parent_method) and parent_method.__doc__:
                        # Copy the docstring from the specified parent method
                        attr.__doc__ = parent_method.__doc__
                        break
            continue

        # Check if the method already has a docstring
        if attr.__doc__:
            continue

        # Automatic inheritance: Look for docstring in parent classes with same method name
        for base in cls.__mro__[1:]:  # Skip the class itself
            if hasattr(base, attr_name):
                parent_method = getattr(base, attr_name)
                if callable(parent_method) and parent_method.__doc__:
                    # Copy the docstring
                    attr.__doc__ = parent_method.__doc__
                    break

    return cls


def inherit_docstring_from_parent(method_name: str = None):
    r"""
    Method decorator that inherits docstring from a specific parent class method.

    This decorator allows you to explicitly inherit a docstring from a parent class method,
    even if the method names are different.

    Usage:

        .. code-block:: python

            class ChildClass(ParentClass):
                @inherit_docstring_from_parent('parent_method_name')
                def child_method(self):
                    pass

                @inherit_docstring_from_parent()  # Uses same method name
                def some_method(self):
                    pass

    Args:
        method_name: Name of the parent method to inherit docstring from.
                    If None, uses the decorated method's name.

    Returns:
        The decorated method with inherited docstring
    """

    def decorator(func):
        target_method_name = method_name or func.__name__

        # We need to defer the docstring inheritance until the class is fully defined
        # So we'll mark the function and handle it in the class decorator
        func._inherit_docstring_from = target_method_name
        return func

    return decorator


if __name__ == "__main__":
    pass
