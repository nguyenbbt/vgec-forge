import argparse
import importlib.metadata
import json
import logging
import os
import platform
import sys
from contextlib import contextmanager
from pathlib import Path

from .config import load_config
from .data import exact_keys, read_jsonl, text_value, validate_dataset, write_json, write_jsonl
from .evaluation import evaluate, index_gold, mine_hard_examples
from .models import make_backend
from .pipeline import Pipeline, validate_verdict
from .prompts import classify_prompt, correction_prompt, eligible_examples


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".run.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ValueError(f"Run is locked: {path}. After confirming no process is active, remove the stale lock manually.") from exc
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def fresh_output(path):
    if Path(path).exists():
        raise ValueError(f"Output exists; choose another path: {path}")


def predict(rows, model_config):
    index_gold(rows)
    examples = eligible_examples([text for row in rows for text in
                                  (row["Input"], row["Output"]["Fixed Sentence"][0])])
    model = make_backend(model_config)
    output = []
    try:
        for row in rows:
            verdict = validate_verdict(model.complete(classify_prompt(row["Input"], examples)))
            if verdict["label"] == 0:
                fixed = row["Input"]
            else:
                answer = model.complete(correction_prompt(row["Input"]))
                exact_keys(answer, ("fixed",))
                fixed = text_value(answer["fixed"])
            output.append({"ID": row["ID"], "label": verdict["label"], "fixed": fixed,
                           "confidence": verdict["confidence"]})
            logging.info("Predicted %s/%s", len(output), len(rows))
    finally:
        model.close()
    return output


def parser():
    result = argparse.ArgumentParser(description="VGEC: Qwen generation, independent verification, evaluation")
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("run", "generate", "verify", "export", "status", "doctor"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/api.json")
    command = sub.add_parser("validate")
    command.add_argument("path")
    command.add_argument("--count", type=int, default=300)
    command.add_argument("--start-id", type=int, default=120001)
    command = sub.add_parser("predict")
    command.add_argument("--config", default="configs/api.json")
    command.add_argument("--data", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--role", choices=("generator", "verifier", "supplementary"), default="verifier")
    command = sub.add_parser("evaluate")
    command.add_argument("--gold", required=True)
    command.add_argument("--predictions", required=True)
    command.add_argument("--scorer-config")
    command.add_argument("--output", required=True)
    command = sub.add_parser("mine")
    command.add_argument("--train", required=True)
    command.add_argument("--predictions", required=True)
    command.add_argument("--holdout", action="append", default=[])
    command.add_argument("--limit", type=int, default=100)
    command.add_argument("--output", required=True)
    command = sub.add_parser("train-classifier")
    command.add_argument("--train", required=True)
    command.add_argument("--dev", required=True)
    command.add_argument("--output-dir", required=True)
    command.add_argument("--folds", type=int, default=5)
    command.add_argument("--seed", type=int, default=42)
    return result


def dispatch(args):
    if args.command == "validate":
        path = Path(args.path)
        rows = json.loads(path.read_text(encoding="utf-8-sig")) if path.suffix == ".json" else read_jsonl(path)
        validate_dataset(rows, args.count, args.start_id)
        return {"valid": True, "count": len(rows), "first_id": args.start_id, "last_id": args.start_id + len(rows) - 1}
    if args.command == "evaluate":
        fresh_output(args.output)
        scorer = None
        if args.scorer_config:
            path = Path(args.scorer_config).resolve()
            scorer = json.loads(path.read_text(encoding="utf-8-sig"))
            scorer["gold_reference"] = str((path.parent / scorer["gold_reference"]).resolve())
            scorer["cwd"] = str((path.parent / scorer.get("cwd", ".")).resolve())
        report = evaluate(read_jsonl(args.gold), read_jsonl(args.predictions), scorer)
        write_json(args.output, report)
        return report
    if args.command == "mine":
        fresh_output(args.output)
        holdout = [row for path in args.holdout for row in read_jsonl(path)]
        rows = mine_hard_examples(read_jsonl(args.train), read_jsonl(args.predictions), holdout, args.limit)
        write_jsonl(args.output, rows)
        return {"review_required": True, "count": len(rows), "path": args.output}
    if args.command == "train-classifier":
        from .classifier import train_baseline
        return train_baseline(read_jsonl(args.train), read_jsonl(args.dev), args.output_dir, args.seed, args.folds)
    config = load_config(args.config)
    if args.command == "doctor":
        versions = {}
        for package in ("torch", "transformers", "accelerate", "bitsandbytes", "scikit-learn"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        return {"python": platform.python_version(), "executable": sys.executable,
                "generator": config["generator"]["model"], "verifier": config["verifier"]["model"],
                "packages": versions, "run_dir": config["run_dir"],
                "note": "Offline inspection only; model weights and endpoints not tested"}
    if args.command == "predict":
        fresh_output(args.output)
        if config[args.role] is None:
            raise ValueError("Selected model role is not configured")
        rows = predict(read_jsonl(args.data), config[args.role])
        write_jsonl(args.output, rows)
        return {"count": len(rows), "path": args.output}
    with run_lock(config["run_dir"]):
        pipeline = Pipeline(config)
        if args.command == "run":
            return {"dataset": str(pipeline.run()), **pipeline.report()}
        if args.command == "generate":
            pipeline.generate_round()
        elif args.command == "verify":
            pipeline.verify()
        elif args.command == "export":
            pipeline.export()
        return pipeline.report()


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, OSError, RuntimeError, KeyError, ImportError) as exc:
        logging.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logging.warning("Interrupted; pipeline checkpoint retained")
        return 130
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
