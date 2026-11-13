import math
from collections.abc import Mapping
from numbers import Real
from typing import Any


def _finite_metrics(data: Mapping[str, Any]) -> dict[str, float]:
    """Keep finite numeric values from a reported payload.

    Args:
        data: values to record.

    Returns:
        Metric names mapped to finite floats. Booleans and non-numeric values
        are omitted.
    """
    metrics: dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        number = float(value)
        if math.isfinite(number):
            metrics[str(key)] = number
    return metrics


def _plain_data(value: Any) -> Any:
    """Convert a config value into data MLflow and wandb can store.

    Args:
        value: a config value, possibly nested.

    Returns:
        Scalars stay as they are. Mappings become dicts with string keys, and
        sequences become lists. Anything else becomes a string.
    """
    if isinstance(value, bool) or value is None or isinstance(value, (str, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_data(item) for item in value]
    return str(value)
