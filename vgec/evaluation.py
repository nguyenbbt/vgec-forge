import json
import math
import subprocess
import tempfile
from pathlib import Path

from .data import Deduplicator, text_value, validate_record, write_jsonl


def index_gold(rows):
    result = {}
    for row in rows:
        validate_record(row)
        if row["ID"] in result:
            raise ValueError("Duplicate gold ID")
        result[row["ID"]] = row
    if not result:
        raise ValueError("Gold dataset is empty")
    return result


def align_predictions(gold, predictions):
    expected = index_gold(gold)
    actual = {}
    for row in predictions:
        if not isinstance(row, dict) or not {"ID", "label"} <= set(row) or set(row) - {"ID", "label", "fixed", "confidence"}:
            raise ValueError("Prediction keys: ID, label, optional fixed and confidence")
        if type(row["ID"]) is not int or type(row["label"]) is not int or row["label"] not in range(5):
            raise ValueError("Prediction ID/label must be integers; label in 0..4")
        if row["ID"] in actual:
            raise ValueError("Duplicate prediction ID")
        if "fixed" in row:
            text_value(row["fixed"], "prediction.fixed")
        if "confidence" in row and (type(row["confidence"]) not in (int, float) or not 0 <= row["confidence"] <= 1):
            raise ValueError("Invalid prediction confidence")
        actual[row["ID"]] = row
    if set(expected) != set(actual):
        raise ValueError("Gold and prediction ID sets must match exactly")
    return [actual[row["ID"]] for row in gold]


def macro_f1(truth, predicted):
    if not truth or len(truth) != len(predicted):
        raise ValueError("Labels must be nonempty and aligned")
    if any(type(x) is not int or x not in range(5) for x in [*truth, *predicted]):
        raise ValueError("Labels must be integer 0..4")
    matrix = [[0] * 5 for _ in range(5)]
    for gold, guess in zip(truth, predicted):
        matrix[gold][guess] += 1
    per_class = []
    for label in range(5):
        tp = matrix[label][label]
        support = sum(matrix[label])
        predicted_count = sum(row[label] for row in matrix)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted_count) if support + predicted_count else 0.0
        per_class.append({"label": label, "precision": precision, "recall": recall, "f1": f1, "support": support})
    return {"macro_f1": sum(row["f1"] for row in per_class) / 5,
            "labels": list(range(5)), "per_class": per_class, "confusion_matrix": matrix,
            "zero_division": 0, "missing_gold_labels": [row["label"] for row in per_class if not row["support"]]}


def run_gec_scorer(gold, aligned, config):
    allowed = {"name", "metric_key", "gold_reference", "command", "timeout", "cwd"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Invalid scorer configuration")
    for key in ("name", "metric_key", "gold_reference"):
        text_value(config.get(key), key)
    command = config.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError("Scorer command must be an argv array, never a shell string")
    if not any("{predictions}" in part for part in command) or not any("{gold}" in part for part in command):
        raise ValueError("Scorer command must include {predictions} and {gold}")
    reference = Path(config["gold_reference"]).resolve(strict=True)
    if any("fixed" not in row for row in aligned):
        raise ValueError("GEC evaluation requires fixed text for every prediction")
    with tempfile.TemporaryDirectory(prefix="vgec-eval-") as directory:
        root = Path(directory)
        predictions_path = root / "predictions.txt"
        predictions_path.write_text("".join(row["fixed"] + "\n" for row in aligned), encoding="utf-8")
        sources = root / "sources.txt"
        sources.write_text("".join(row["Input"] + "\n" for row in gold), encoding="utf-8")
        gold_jsonl = root / "gold.jsonl"
        write_jsonl(gold_jsonl, gold)
        substitutions = {"predictions": str(predictions_path), "gold": str(reference),
                         "sources": str(sources), "gold_jsonl": str(gold_jsonl)}
        args = [part.format_map(substitutions) for part in command]
        # This is operator-owned config. Model output can never supply executable code.
        result = subprocess.run(args, shell=False, capture_output=True, text=True, encoding="utf-8",
                                timeout=config.get("timeout", 300), cwd=config.get("cwd"))
        if result.returncode:
            raise ValueError(f"GEC scorer failed (exit {result.returncode}): {result.stderr[-1500:]}")
        metrics = json.loads(result.stdout)
    if not isinstance(metrics, dict) or config["metric_key"] not in metrics:
        raise ValueError("Scorer must print a JSON object including the configured metric_key")
    if not metrics or any(type(v) not in (int, float) or not math.isfinite(v) for v in metrics.values()):
        raise ValueError("Scorer metrics must all be finite numbers")
    return {"name": config["name"], "metric_key": config["metric_key"], "metrics": metrics,
            "gold_reference": str(reference), "command_template": command}


def evaluate(gold, predictions, scorer=None):
    aligned = align_predictions(gold, predictions)
    report = {"count": len(gold), "classifier": macro_f1(
        [row["Output"]["Error_Type_Id"] for row in gold], [row["label"] for row in aligned]),
        "gec_official": None, "gec_status": "not_configured"}
    if scorer is not None:
        report["gec_official"] = run_gec_scorer(gold, aligned, scorer)
        report["gec_status"] = "external_scorer_completed"
    return report


def assert_no_overlap(train, holdout, threshold=0.9):
    excluded = Deduplicator(threshold)
    for row in holdout:
        validate_record(row)
        excluded.add(row["Input"])
        excluded.add(row["Output"]["Fixed Sentence"][0])
    for row in train:
        if excluded.contains(row["Input"]) or excluded.contains(row["Output"]["Fixed Sentence"][0]):
            raise ValueError("Train/hard-mining data overlaps holdout")


def mine_hard_examples(gold, predictions, holdout=(), limit=100, uncertain_below=0.7):
    if type(limit) is not int or limit < 1:
        raise ValueError("Mining limit must be positive")
    aligned = align_predictions(gold, predictions)
    assert_no_overlap(gold, holdout)
    candidates = []
    for row, guess in zip(gold, aligned):
        wrong = guess["label"] != row["Output"]["Error_Type_Id"]
        confidence = guess.get("confidence", 1.0)
        if not wrong and confidence >= uncertain_below:
            continue
        candidates.append({"record": row, "split": "train", "human_verified": False,
                           "predicted_label": guess["label"], "confidence": confidence,
                           "hard_reason": "classifier_error" if wrong else "classifier_uncertain",
                           "priority": 1 + confidence if wrong else 1 - confidence})
    return sorted(candidates, key=lambda x: (-x["priority"], x["record"]["ID"]))[:limit]
