"""Sequential stages keep only one local model resident at a time."""

import json
import logging
import math
from collections import Counter, defaultdict, deque
from importlib.resources import files
from pathlib import Path

from .config import validate_config
from .data import (Deduplicator, digest, exact_keys, function_only_edit, normalize,
                   read_jsonl, read_records, references, text_value, validate_dataset,
                   validate_gold_record, write_json, write_jsonl)
from .models import make_backend
from .prompts import (CONTROLLED_PATTERNS, EXAMPLES, RUBRIC, SUBTYPES, TOPICS,
                      audit_prompt, classify_prompt, controlled_seed_prompt,
                      eligible_examples, injection_prompt, seed_prompt)

LOG = logging.getLogger(__name__)


def validate_candidate(value, clean):
    exact_keys(value, ("input", "error_span", "fixed_span", "explanation"))
    for key in value:
        text_value(value[key], key)
    source = value["input"]
    error, fixed = value["error_span"], value["fixed_span"]
    if normalize(source) == normalize(clean):
        raise ValueError("unchanged_sentence")
    start = 0
    while (index := source.find(error, start)) >= 0:
        if source[:index] + fixed + source[index + len(error):] == clean:
            break
        start = index + 1
    else:
        raise ValueError("nonlocal_or_inexact_repair")
    if function_only_edit(source, clean):
        raise ValueError("function_or_punctuation_only_heuristic")
    return value


def controlled_candidate(clean, pattern):
    fixed = pattern["fixed_span"]
    if clean.count(fixed) != 1:
        raise ValueError("missing_or_repeated_required_phrase")
    source = clean.replace(fixed, pattern["error_span"], 1)
    return validate_candidate({"input": source, "error_span": pattern["error_span"],
                               "fixed_span": fixed,
                               "explanation": pattern["explanation"]}, clean)


def validate_verdict(value):
    exact_keys(value, ("label", "confidence", "reason"))
    if type(value["label"]) is not int or value["label"] not in range(5):
        raise ValueError("Invalid verifier label")
    confidence = value["confidence"]
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("Invalid verifier confidence")
    text_value(value["reason"], "reason")
    return value


def validate_audit(value):
    booleans = ("grammatical_fixed", "meaning_preserved", "minimal", "phrase_only",
                "function_only", "punctuation_only")
    exact_keys(value, (*booleans, "error_count", "reason"))
    if any(type(value[key]) is not bool for key in booleans):
        raise ValueError("Audit flags must be booleans")
    if type(value["error_count"]) is not int or value["error_count"] < 0:
        raise ValueError("Audit error_count must be a nonnegative integer")
    text_value(value["reason"], "reason")
    return value


class Pipeline:
    def __init__(self, config, factory=make_backend):
        self.config = validate_config(config)
        self.factory = factory
        self.directory = Path(self.config["run_dir"])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint = self.directory / "checkpoint.json"
        self.excluded = Deduplicator(self.config["dedup_threshold"])
        fingerprints = {}
        for path in [*self.config["holdout_paths"], self.config["seed_path"], self.config["hard_examples_path"]]:
            if path:
                fingerprints[str(path)] = digest(Path(path).read_text(encoding="utf-8-sig"))
        holdout_texts = []
        for path in self.config["holdout_paths"]:
            for row in read_records(path):
                for text in [row["Input"], *references(row)]:
                    self.excluded.add(text)
                    holdout_texts.append(text)
        self.examples = eligible_examples(holdout_texts, threshold=self.config["dedup_threshold"])
        for row in EXAMPLES:
            self.excluded.add(row["Input"])
            self.excluded.add(row["fixed"])
        self.hard_examples = self._hard_examples()
        self.seeds = None
        if self.config["seed_path"]:
            self.seeds = read_jsonl(self.config["seed_path"])
            if not self.seeds:
                raise ValueError("Seed corpus is empty")
            for row in self.seeds:
                exact_keys(row, ("sentence",))
                text_value(row["sentence"])
        # Budget may be increased on resume; quality, models and corpus may not change.
        signature_config = {k: v for k, v in self.config.items() if k != "max_rounds"}
        signature = digest({"config": signature_config, "files": fingerprints,
                            "rubric": RUBRIC, "examples": EXAMPLES, "version": 1,
                            "prompts": files("vgec").joinpath("prompts.py").read_text(encoding="utf-8")})
        if self.checkpoint.exists():
            self.state = json.loads(self.checkpoint.read_text(encoding="utf-8"))
            if self.state["signature"] != signature:
                raise ValueError("Run configuration or inputs changed; choose a new run_dir")
        else:
            if any((self.directory / name).exists() for name in ("dataset.jsonl", "dataset.json", "audit.jsonl")):
                raise ValueError("Existing output without matching checkpoint; choose a new run_dir")
            self.state = {"signature": signature, "config": signature_config, "attempts": 0,
                          "candidates": [], "decisions": [], "generation_rejections": []}
            self._save()

    def _hard_examples(self):
        path = self.config["hard_examples_path"]
        if not path:
            return []
        result = []
        for item in read_jsonl(path):
            if item.get("split") != "train" or item.get("human_verified") is not True:
                raise ValueError("Hard examples require train provenance and human_verified=true")
            row = validate_gold_record(item["record"])
            if row["Output"]["Error_Type_Id"] != 2:
                continue
            fixed = references(row)[0]
            if any(self.excluded.contains(text) for text in [row["Input"], *references(row)]):
                raise ValueError("Hard example overlaps holdout or built-in few-shot")
            result.append({"Input": row["Input"], "label": 2, "fixed": fixed,
                           "reason": "Gold train example misclassified by the baseline; retain gold label."})
            self.excluded.add(row["Input"])
            for text in references(row):
                self.excluded.add(text)
        return result

    def _save(self):
        write_json(self.checkpoint, self.state)

    def accepted(self):
        return [d for d in self.state["decisions"] if d["status"] == "accepted"]

    def generate_round(self):
        cfg = self.config
        attempts = self.state["attempts"]
        round_index = attempts // cfg["round_size"]
        if round_index >= cfg["max_rounds"]:
            raise ValueError("Generation budget exhausted; increase max_rounds or inspect rejections")
        if self.seeds is not None and attempts >= len(self.seeds):
            raise ValueError("Seed corpus exhausted; use a larger corpus in a new run or automatic seeds")
        model_cfg = cfg["generator"]
        if cfg["supplementary"] and round_index >= cfg["fallback_after_round"] and round_index % 2 == 1:
            model_cfg = cfg["supplementary"]
        model = self.factory(model_cfg)
        seen = Deduplicator(cfg["dedup_threshold"])
        for item in self.state["candidates"]:
            seen.add(item["input"])
            seen.add(item["fixed"])
        stop = (round_index + 1) * cfg["round_size"]
        if self.seeds is not None:
            stop = min(stop, len(self.seeds))
        try:
            while self.state["attempts"] < stop:
                index = self.state["attempts"]
                pattern = CONTROLLED_PATTERNS[index % len(CONTROLLED_PATTERNS)] if cfg["controlled_injection"] else None
                subtype = pattern["subtype"] if pattern else list(SUBTYPES)[index % len(SUBTYPES)]
                topic = TOPICS[(index // len(SUBTYPES) + index) % len(TOPICS)]
                seed = cfg["seed"] + index * 10
                try:
                    if self.seeds is not None:
                        clean = self.seeds[index]["sentence"]
                    else:
                        prompt = (controlled_seed_prompt(topic, pattern, seed) if pattern else
                                  seed_prompt(topic, subtype, seed))
                        response = model.complete(prompt, temperature=0.8, seed=seed)
                        exact_keys(response, ("sentence",))
                        clean = text_value(response["sentence"])
                    if self.excluded.contains(clean) or seen.contains(clean):
                        raise ValueError("duplicate_or_holdout_seed")
                    if pattern:
                        value = controlled_candidate(clean, pattern)
                    else:
                        hard = self.hard_examples[index % len(self.hard_examples):][:2] if self.hard_examples else []
                        response = model.complete(injection_prompt(clean, subtype, hard, self.examples), temperature=0.7, seed=seed + 1)
                        value = validate_candidate(response, clean)
                    if self.excluded.contains(value["input"]) or seen.contains(value["input"]):
                        raise ValueError("duplicate_or_holdout_input")
                    item = {**value, "fixed": clean, "cid": digest([value["input"], clean]),
                            "subtype": subtype, "topic": topic, "generator": model_cfg["model"]}
                    self.state["candidates"].append(item)
                    seen.add(clean)
                    seen.add(value["input"])
                except ValueError as exc:
                    self.state["generation_rejections"].append({"attempt": index, "reason": str(exc)[:300]})
                self.state["attempts"] += 1
                self._save()
                LOG.info("Generation attempts=%s candidates=%s", self.state["attempts"], len(self.state["candidates"]))
        finally:
            model.close()

    def verify(self):
        done = {d["cid"] for d in self.state["decisions"]}
        pending = [c for c in self.state["candidates"] if c["cid"] not in done]
        if not pending:
            return
        model = self.factory(self.config["verifier"])
        try:
            for item in pending:
                decision = {"cid": item["cid"], "status": "rejected", "reason": "label_disagreement"}
                try:
                    blind = validate_verdict(model.complete(classify_prompt(item["input"], self.examples)))
                    decision["blind"] = blind
                    if blind["label"] == 2 and blind["confidence"] >= self.config["min_confidence"]:
                        audit = validate_audit(model.complete(audit_prompt(item["input"], item["fixed"])))
                        decision["audit"] = audit
                        good = (all(audit[k] for k in ("grammatical_fixed", "meaning_preserved", "minimal", "phrase_only"))
                                and not audit["function_only"] and not audit["punctuation_only"] and audit["error_count"] == 1)
                        decision["status"] = "accepted" if good else "rejected"
                        decision["reason"] = "passed" if good else "pair_audit_failed"
                except ValueError as exc:
                    decision["reason"] = "invalid_verifier_response: " + str(exc)[:250]
                self.state["decisions"].append(decision)
                self._save()
                LOG.info("Verified=%s accepted=%s", len(self.state["decisions"]), len(self.accepted()))
        finally:
            model.close()

    def export(self):
        decisions = {d["cid"]: d for d in self.accepted()}
        groups = defaultdict(deque)
        for item in self.state["candidates"]:
            if item["cid"] in decisions:
                groups[item["subtype"]].append(item)
        chosen = []
        while groups and len(chosen) < self.config["count"]:
            for key in list(groups):
                if len(chosen) < self.config["count"]:
                    chosen.append(groups[key].popleft())
                if not groups[key]:
                    del groups[key]
        rows = [{"ID": self.config["start_id"] + index, "Input": item["input"],
                 "Output": {"Error_Type_Id": 2, "Fixed Sentence": [item["fixed"]]}}
                for index, item in enumerate(chosen)]
        validate_dataset(rows, self.config["count"], self.config["start_id"])
        target = self.directory / "dataset.jsonl"
        audit_rows = [{"ID": row["ID"], "candidate": item, "verification": decisions[item["cid"]]}
                      for row, item in zip(rows, chosen)]
        # Preflight all final artifacts before writing any of them.
        for name, expected in (("dataset.jsonl", rows), ("dataset.json", rows), ("audit.jsonl", audit_rows)):
            path = self.directory / name
            if path.exists():
                existing = json.loads(path.read_text(encoding="utf-8")) if name.endswith(".json") else read_jsonl(path)
                if existing != expected:
                    raise ValueError(f"Refusing to replace an existing, different final artifact: {name}")
        write_jsonl(target, rows)
        write_json(self.directory / "dataset.json", rows)
        write_jsonl(self.directory / "audit.jsonl", audit_rows)
        self.report()
        return target

    def report(self):
        report = {"attempts": self.state["attempts"], "candidates": len(self.state["candidates"]),
                  "verified": len(self.state["decisions"]), "accepted": len(self.accepted()),
                  "target": self.config["count"], "rejections": dict(Counter(
                      d["reason"] for d in self.state["decisions"] if d["status"] != "accepted")),
                  "blind_labels": dict(Counter(
                      str(d["blind"]["label"]) for d in self.state["decisions"] if "blind" in d)),
                  "generation_rejected": len(self.state["generation_rejections"]),
                  "confidence_note": "Model self-reported; not calibrated probability"}
        write_json(self.directory / "report.json", report)
        return report

    def run(self):
        self.verify()
        while len(self.accepted()) < self.config["count"]:
            if self.state["attempts"] >= self.config["round_size"] * self.config["max_rounds"]:
                self.report()
                raise ValueError(f"Only {len(self.accepted())}/{self.config['count']} accepted; inspect report and increase max_rounds")
            self.generate_round()
            self.verify()
            self.report()
        return self.export()
