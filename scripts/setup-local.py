"""Finish runtime setup in the background and leave machine-readable evidence."""
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vgec.data import write_json


def main():
    status = {'state': 'installing', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'python': sys.executable, 'model_weights_downloaded': False}
    path = ROOT/'outputs/local-setup/status.json'
    write_json(path, status)
    try:
        subprocess.run([sys.executable, '-B', str(ROOT/'scripts/download-runtime.py')], cwd=ROOT, check=True)
        import torch
        import transformers
        import bitsandbytes as bnb
        from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        layer = bnb.nn.Linear4bit(64, 64, bias=False, quant_type='nf4').to('cuda')
        with torch.inference_mode():
            output = layer(torch.randn(1, 64, device='cuda', dtype=torch.float16))
        if not torch.isfinite(output).all():
            raise RuntimeError('4-bit CUDA smoke returned nonfinite output')
        status.update(state='runtime_ready', torch=torch.__version__, transformers=transformers.__version__,
                      bitsandbytes=bnb.__version__, gpu=torch.cuda.get_device_name(),
                      cuda_4bit_smoke='passed', model_inference_smoke='not_run')
    except Exception as exc:
        status.update(state='failed', error=str(exc), traceback=traceback.format_exc())
        print(status['traceback'], file=sys.stderr)
    finally:
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(path, status)
    print(json.dumps(status, ensure_ascii=True, indent=2), flush=True)
    return 0 if status['state'] == 'runtime_ready' else 1


if __name__ == '__main__':
    raise SystemExit(main())
