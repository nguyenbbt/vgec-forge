# Verification record

Implemented against SPEC.md; no real model inference or benchmark scores claimed.

- Core and CLI run with Python 3.12.7, no third-party packages required.
- Unittest covers strict output contracts, 300 consecutive IDs, independent
  verification gates, corpus exhaustion, resume, near-duplicate filtering,
  holdout/few-shot leakage, HTTP transport and external scorer subprocess.
- A separate review examined correctness, security, architecture and tests.
  Findings addressed with failing-before/passing-after regressions:
  heldout examples in prompts, all-artifact overwrite checks, label-0 prediction
  preservation, and local Qwen3.5 dispatch by AutoConfig rather than filename.
- Optional Transformers and sklearn execution still requires dependencies;
  Transformers also requires model weights. Runtime quality and GPU fit remain unmeasured.
- Official VGEC metric remains user-supplied: no benchmark name, dev corpus or
  scorer was present in the workspace.
- Default thresholds and baseline are engineering starting points, not results
  calibrated on a real dev set. Inspect sampled audit decisions before training.
