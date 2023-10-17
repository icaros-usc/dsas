"""Functions for managing worker state.

In general, one uses these by first calling init_* or set_* to create the
attribute, then calling get_* to retrieve the corresponding value.
"""
from dask.distributed import get_worker

#
# Generic
#


def set_worker_state(key: str, val: object):
    """Sets worker_state[key] = val"""
    worker = get_worker()
    setattr(worker, key, val)


def get_worker_state(key: str) -> object:
    """Retrieves worker_state[key]"""
    worker = get_worker()
    return getattr(worker, key)
