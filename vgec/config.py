import copy
import json
import math
from pathlib import Path


DEFAULTS = {
    "run_dir": "outputs/vgec-label2", "count": 300, "start_id": 120001,
    "seed": 42, "round_size": 60, "max_rounds": 40,
    "min_confidence": 0.85, "dedup_threshold": 0.9, "fallback_after_round": 3,
    "seed_path": None, "hard_examples_path": None, "holdout_paths": [],
    "generator": {"backend": "api", "model": "Qwen/Qwen3.5-4B",
                  "base_url": "http://localhost:8000/v1", "max_tokens": 2048,
                  "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    "supplementary": None,
    "verifier": {"backend": "api", "model": "Qwen/Qwen3-4B-Instruct-2507",
                 "base_url": "http://localhost:8001/v1", "max_tokens": 2048},
}
MODEL_KEYS = {"backend", "model", "base_url", "max_tokens", "extra_body", "api_key_env",
              "timeout", "retries", "top_p", "load_in_4bit", "device_map", "context_limit", "revision"}


def validate_config(raw):
    if not isinstance(raw, dict) or set(raw) - set(DEFAULTS):
        raise ValueError("Unknown or invalid config keys")
    config = copy.deepcopy(DEFAULTS)
    for key, value in raw.items():
        if key in ("generator", "verifier") and isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    for key in ("count", "start_id", "round_size", "max_rounds", "fallback_after_round"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    for key in ("min_confidence", "dedup_threshold"):
        if type(config[key]) not in (int, float) or not 0 < config[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    for role in ("generator", "verifier", "supplementary"):
        model = config[role]
        if model is None and role == "supplementary":
            continue
        if not isinstance(model, dict) or set(model) - MODEL_KEYS:
            raise ValueError(f"Unknown {role} configuration; keys belong in environment variables")
        if not isinstance(model.get("model"), str) or not model["model"].strip():
            raise ValueError(f"{role}.model must be a model identifier")
        if model.get("backend", "api") not in ("api", "transformers"):
            raise ValueError("backend must be api or transformers")
        for key in ("max_tokens", "context_limit"):
            value = model.get(key, 1)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{role}.{key} must be a positive integer")
        for key in ("timeout",):
            value = model.get(key, 1)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {role}.{key}")
        retries = model.get("retries", 2)
        if type(retries) is not int or not 0 <= retries <= 10:
            raise ValueError("retries must be an integer in 0..10")
        if "load_in_4bit" in model and type(model["load_in_4bit"]) is not bool:
            raise ValueError("load_in_4bit must be boolean")
        if model.get("backend", "api") == "api" and not isinstance(model.get("base_url"), str):
            raise ValueError(f"{role}.base_url required for API backend")
        if not isinstance(model.get("extra_body", {}), dict):
            raise ValueError("extra_body must be an object")
        extra = model.get("extra_body", {})
        if set(extra) - {"top_k", "min_p", "presence_penalty", "repetition_penalty",
                         "chat_template_kwargs", "enable_thinking", "response_format"}:
            raise ValueError("extra_body accepts only generation settings, never credentials")
        top_p = model.get("top_p", 0.8)
        if type(top_p) not in (float, int) or not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
    for role in ("generator", "supplementary"):
        if config[role] and config[role]["model"] == config["verifier"]["model"]:
            raise ValueError("Independent verification requires a different model identifier")
    if not isinstance(config["holdout_paths"], list) or not all(isinstance(p, str) for p in config["holdout_paths"]):
        raise ValueError("holdout_paths must be an array of paths")
    if not isinstance(config["run_dir"], str) or not config["run_dir"].strip():
        raise ValueError("run_dir must be a nonempty path")
    for key in ("seed_path", "hard_examples_path"):
        if config[key] is not None and (not isinstance(config[key], str) or not config[key].strip()):
            raise ValueError(f"{key} must be null or a nonempty path")
    return config


def load_config(path):
    path = Path(path).resolve()
    config = validate_config(json.loads(path.read_text(encoding="utf-8-sig")))
    for key in ("run_dir", "seed_path", "hard_examples_path"):
        if config[key]:
            config[key] = str((path.parent / config[key]).resolve())
    config["holdout_paths"] = [str((path.parent / p).resolve()) for p in config["holdout_paths"]]
    return config
