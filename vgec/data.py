"""Strict interchange contracts. Normalization is for comparison only."""

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path


def text_value(value, name="text") -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 4000:
        raise ValueError(f"{name}: expected nonempty text, at most 4000 characters")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{name}: expected a single line")
    return value


def exact_keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"Expected exactly keys: {', '.join(keys)}")


def validate_record(row):
    exact_keys(row, ("ID", "Input", "Output"))
    if type(row["ID"]) is not int or row["ID"] < 0:
        raise ValueError("ID must be a nonnegative integer")
    text_value(row["Input"], "Input")
    output = row["Output"]
    exact_keys(output, ("Error_Type_Id", "Fixed Sentence"))
    label = output["Error_Type_Id"]
    if type(label) is not int or label not in range(5):
        raise ValueError("Error_Type_Id must be integer 0..4")
    fixed = output["Fixed Sentence"]
    if not isinstance(fixed, list) or len(fixed) != 1:
        raise ValueError("Fixed Sentence must be a one-element array")
    text_value(fixed[0], "Fixed Sentence")
    if label == 0 and fixed[0] != row["Input"]:
        raise ValueError("Label 0 must preserve Input byte-for-byte after JSON decoding")
    if label != 0 and fixed[0] == row["Input"]:
        raise ValueError("An error label requires a changed correction")
    return row


def validate_dataset(rows, count=300, start_id=120001):
    if len(rows) != count:
        raise ValueError(f"Expected {count} records, got {len(rows)}")
    inputs, corrections = set(), set()
    for expected_id, row in enumerate(rows, start_id):
        validate_record(row)
        if row["ID"] != expected_id or row["Output"]["Error_Type_Id"] != 2:
            raise ValueError("Export requires consecutive IDs and integer label 2")
        source, fixed = normalize(row["Input"]), normalize(row["Output"]["Fixed Sentence"][0])
        if source in inputs or fixed in corrections:
            raise ValueError("Duplicate Input or correction")
        inputs.add(source)
        corrections.add(fixed)
    return rows


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_jsonl(path, rows):
    atomic_text(path, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


class Deduplicator:
    """Exact + character sequence similarity; suitable for a pool of a few thousand."""

    def __init__(self, threshold=0.9):
        if not 0 < threshold <= 1:
            raise ValueError("Dedup threshold must be in (0, 1]")
        self.threshold = threshold
        self.values = set()

    def contains(self, text):
        value = normalize(text)
        if value in self.values:
            return True
        for previous in self.values:
            # Upper bound avoids expensive comparisons when lengths differ greatly.
            if 2 * min(len(value), len(previous)) / max(1, len(value) + len(previous)) < self.threshold:
                continue
            if SequenceMatcher(None, value, previous, autojunk=False).ratio() >= self.threshold:
                return True
        return False

    def add(self, text):
        self.values.add(normalize(text))


FUNCTION_WORDS = set("của với qua bằng trong giữa vì và nhưng tuy nhiên là đã đang sẽ được bị không cũng còn chỉ mà thì do bởi để từ tới đến cho về ở vào ra nên nếu dù tuy rất quá lắm những các một này đó ấy".split())


def function_only_edit(source, fixed):
    """Conservative rejection heuristic, never a proof of grammatical label."""
    left = re.findall(r"\w+|[^\w\s]", normalize(source))
    right = re.findall(r"\w+|[^\w\s]", normalize(fixed))
    changed = []
    for op, a, b, c, d in SequenceMatcher(None, left, right, autojunk=False).get_opcodes():
        if op != "equal":
            changed.extend(left[a:b] + right[c:d])
    return bool(changed) and all(token in FUNCTION_WORDS or not re.search(r"\w", token) for token in changed)
