"""Count-specific settings for the Modal runner."""

from copy import deepcopy

from .config import validate_config

RUN_PREFIX = "vgec-label2-modal-v4"


def run_name(count):
    if type(count) is not int or count < 1:
        raise ValueError("--count must be a positive integer")
    return f"{RUN_PREFIX}-{count}"


def config_for_count(base, count):
    name = run_name(count)
    config = deepcopy(base)
    config["count"] = count
    config["round_size"] = min(60, max(10, count * 2))
    config["run_dir"] = f"/mnt/runs/{name}"
    return validate_config(config)
