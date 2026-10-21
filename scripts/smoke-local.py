"""Exercise both real models sequentially; write evidence, not a fabricated dataset."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vgec.config import load_config
from vgec.data import write_json
from vgec.models import make_backend
from vgec.pipeline import validate_candidate, validate_verdict, validate_audit
from vgec.prompts import seed_prompt, injection_prompt, classify_prompt, audit_prompt


def main():
    config = load_config('configs/local-4bit.json')
    report = {"purpose": "real local inference smoke; not a quality benchmark", "models": {}}
    generator = make_backend(config['generator'])
    start = time.monotonic()
    try:
        seed = generator.complete(seed_prompt('sinh hoạt gia đình', 'noun_order', 42), temperature=0.7)
        report['seed'] = seed
        print('Seed:', json.dumps(seed, ensure_ascii=True), flush=True)
        candidate = generator.complete(injection_prompt(seed['sentence'], 'noun_order'), temperature=0.7)
        report['candidate'] = candidate
        print('Candidate:', json.dumps(candidate, ensure_ascii=True), flush=True)
        try:
            validate_candidate(candidate, seed['sentence'])
            report['candidate_structurally_valid'] = True
        except ValueError as exc:
            report['candidate_structurally_valid'] = False
            report['candidate_error'] = str(exc)
        report['models']['generator'] = {"seconds": time.monotonic()-start}
    finally:
        generator.close()
        write_json('outputs/local-smoke/report.json', report)
    verifier = make_backend(config['verifier'])
    start = time.monotonic()
    try:
        report['blind'] = validate_verdict(verifier.complete(classify_prompt(candidate['input'])))
        print('Blind:', json.dumps(report['blind'], ensure_ascii=True), flush=True)
        report['clean'] = validate_verdict(verifier.complete(classify_prompt(seed['sentence'])))
        report['audit'] = validate_audit(verifier.complete(audit_prompt(candidate['input'], seed['sentence'])))
        report['models']['verifier'] = {"seconds": time.monotonic()-start}
        import torch
        report['cuda'] = {"available": torch.cuda.is_available(), "name": torch.cuda.get_device_name(),
                          "max_allocated_gib": torch.cuda.max_memory_allocated()/1024**3}
    finally:
        verifier.close()
        write_json('outputs/local-smoke/report.json', report)
    print(json.dumps(report, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
