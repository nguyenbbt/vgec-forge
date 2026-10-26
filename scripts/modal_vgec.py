"""Run the VGEC pipeline on Modal and retrieve its validated JSONL output."""

import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA_FILE = "VGEC 2026_v2.json"
REMOTE_CONFIG = "/app/configs/modal.json"

source_volume = modal.Volume.from_name("vgec-source", create_if_missing=True)
run_volume = modal.Volume.from_name("vgec-runs", create_if_missing=True)
cache_volume = modal.Volume.from_name("vgec-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.10.0", "transformers==5.20.2", "accelerate==1.12.0",
        "bitsandbytes==0.50.2", "Pillow>=10,<13",
    )
    .env({"PYTHONPATH": "/app", "HF_HOME": "/mnt/cache",
          "TOKENIZERS_PARALLELISM": "false"})
    # Modal requires local files to be added after all image build steps.
    .add_local_dir(ROOT / "vgec", remote_path="/app/vgec")
    .add_local_file(ROOT / "configs/modal.json", remote_path=REMOTE_CONFIG)
)
app = modal.App("vgec-factory")


@app.function(
    image=image,
    gpu="A100-40GB",
    max_containers=1,
    timeout=86400,
    volumes={"/mnt/data": source_volume.with_mount_options(read_only=True),
             "/mnt/runs": run_volume, "/mnt/cache": cache_volume},
)
def run_round(count: int):
    """Resume pending verification, generate one round, and commit progress."""
    from vgec.config import load_config
    from vgec.modal_support import config_for_count
    from vgec.pipeline import Pipeline

    run_volume.reload()
    cache_volume.reload()
    if not Path("/mnt/data", DATA_FILE).is_file():
        raise FileNotFoundError(f"Missing Modal corpus: /mnt/data/{DATA_FILE}")
    config = config_for_count(load_config(REMOTE_CONFIG), count)
    try:
        pipeline = Pipeline(config)
        pipeline.verify()
        if len(pipeline.accepted()) < config["count"]:
            pipeline.generate_round()
            pipeline.verify()
        if len(pipeline.accepted()) >= config["count"]:
            pipeline.export()
        return pipeline.report()
    finally:
        run_volume.commit()
        cache_volume.commit()


@app.local_entrypoint()
def main(
    corpus: str = r"D:\Project\VLSP_VGEC_2026\data\VGEC 2026_v2.json",
    count: int = 10,
    output: str = "",
):
    """Upload holdout, run/resume rounds, and download the final dataset."""
    from vgec.data import validate_dataset
    from vgec.modal_support import run_name

    name = run_name(count)
    corpus_path = Path(corpus).resolve(strict=True)
    if not corpus_path.is_file():
        raise ValueError(f"Corpus is not a file: {corpus_path}")
    with source_volume.batch_upload(force=True) as upload:
        upload.put_file(str(corpus_path), f"/{DATA_FILE}")

    while True:
        report = run_round.remote(count)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if report["accepted"] >= report["target"]:
            break

    # read_file yields byte chunks; lines may cross chunk boundaries.
    payload = b"".join(run_volume.read_file(f"/{name}/dataset.jsonl"))
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    validate_dataset(rows, count, 120001)
    destination = Path(output or f"outputs/{name}/dataset.jsonl").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != payload:
        raise ValueError(f"Local output differs; choose another --output: {destination}")
    destination.write_bytes(payload)
    print(f"Saved {len(rows)} verified records to {destination}")
