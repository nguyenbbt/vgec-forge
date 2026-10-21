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

## Local setup follow-up
- User selected Transformers locally and supplied the original 457-row JSON.
- Confirmed external CPython 3.13.2 + torch 2.10.0+cu128 can see the RTX 4050.
  A Gen-local venv shares existing CUDA packages read-only, with upgraded packages
  installed locally. Dependency download is resumable and reports status separately.
- Added JSON-array/multi-reference gold input support without weakening synthetic
  export validation. All references are excluded from generation/few-shot.
- 39 tests pass. Raw corpus SHA-256:
  `44282dca792a6abefcd198b25e962e56739d3374d62d0a587c35ad2c6a30738b`.
- Model download remained incomplete due slow/timeouting network transfers;
  do not infer successful real-model execution from unit tests or CUDA availability.
- The external metric PDF names EM, GLEU+ and ROUGE-L; its accompanying repository
  also marks evaluation provisional and has no official scorer implementation.
