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
        "torch==2.10.0", "torchvision==0.25.0",
        "transformers==5.20.2", "accelerate==1.12.0",
        "bitsandbytes==0.50.2", "Pillow>=10,<13",
    )
    .run_commands("python -c 'import torch, torchvision; print(torch.__version__, torchvision.__version__)'")
    .env({"PYTHONPATH": "/app", "HF_HOME": "/mnt/cache",
          "TOKENIZERS_PARALLELISM": "false"})
    # Modal requires local files to be added after all image build steps.
    .add_local_dir(ROOT / "vgec", remote_path="/app/vgec")
    .add_local_file(ROOT / "configs/modal.json", remote_path=REMOTE_CONFIG)
)
app = modal.App("vgec-factory")


@app.function(image=image, timeout=1800, volumes={"/mnt/cache": cache_volume})
def image_smoke():
    """Load the exact Qwen processor that failed, without allocating a GPU."""
    import torch
    import torchvision
    import transformers
    from vgec.config import load_config

    config = load_config(REMOTE_CONFIG)["generator"]
    try:
        processor = transformers.AutoProcessor.from_pretrained(
            config["model"], revision=config["revision"], trust_remote_code=False)
        result = {"torch": torch.__version__, "torchvision": torchvision.__version__,
                  "transformers": transformers.__version__,
                  "processor": type(processor).__name__}
        print(json.dumps(result), flush=True)
        return result
    finally:
        cache_volume.commit()


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

    if not Path("/mnt/data", DATA_FILE).is_file():
        raise FileNotFoundError(f"Missing Modal corpus: /mnt/data/{DATA_FILE}")
    config = config_for_count(load_config(REMOTE_CONFIG), count)
    try:
        pipeline = Pipeline(config)
        pipeline.verify()
        if len(pipeline.accepted()) < config["count"]:
            before = len(pipeline.state["candidates"])
            pipeline.generate_round()
            pipeline.verify()
            if len(pipeline.state["candidates"]) == before:
                pipeline.report()
                raise ValueError("No valid candidates in this round; inspect checkpoint generation_rejections")
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

    stalled_rounds = 0
    previous_accepted = 0
    while True:
        report = run_round.remote(count)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        if report["accepted"] >= report["target"]:
            break
        stalled_rounds = stalled_rounds + 1 if report["accepted"] == previous_accepted else 0
        previous_accepted = report["accepted"]
        if stalled_rounds >= 2:
            raise RuntimeError("Two rounds without a newly accepted sentence; inspect verifier decisions in the Modal checkpoint")

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
