# Technical Report: VGEC Label-2 Synthetic Data Generation on Modal

## Scope and evidence

- This report describes the completed **10-record Modal run** configured by `configs/modal.json` and executed through `scripts/modal_vgec.py`. Public sample artifacts are in `examples/modal-v4-10/`. It distinguishes code-supported design choices from observed run outcomes.
- Repository evidence: `vgec/prompts.py`, `vgec/pipeline.py`, `vgec/models.py`, `vgec/data.py`, `vgec/config.py`, `vgec/modal_support.py`, and `vgec/resources/fewshot.json`. Run evidence: `report.json`, `generation_rejections.json`, `audit.jsonl`, `dataset.jsonl`, and `parameter_count.json` in the example directory.

## 1. Model specification

- **Actual generator:** `Qwen/Qwen3.5-4B`, pinned to Hugging Face revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The [publisher's model card](https://huggingface.co/Qwen/Qwen3.5-4B) describes a 4B-parameter language model with a vision encoder. This run used text input but loaded the full `Qwen3_5ForConditionalGeneration` checkpoint.
- **Parameter-count verification:** A separate measurement loaded that pinned generator checkpoint without 4-bit quantization and summed `numel()` over unique `named_parameters()`. It found **4,539,265,536 parameters in total**, comprising **4,205,751,296** parameters under names containing `language_model` and **333,514,240** other parameters. The “other” subtotal should not be equated exclusively with the vision encoder without a finer module breakdown. The full loaded checkpoint is therefore **below 8 billion parameters**. The measurement is preserved in `examples/modal-v4-10/parameter_count.json`; the normal generation run used 4-bit loading, which changes representation rather than the model's architectural parameter count.
- **Independent verifier:** `Qwen/Qwen3-4B-Instruct-2507`, pinned to revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Its [publisher's model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) reports **4.0B parameters**. It did not generate the dataset sentences.
- **Supplementary generator:** `supplementary` is `null` in the active Modal configuration. Consequently, `Qwen/Qwen2.5-7B-Instruct` was **not called** in this run.
- **Execution stack:** Hugging Face Transformers **5.20.2** on Modal **A100 40 GB**, Python **3.12**, PyTorch **2.10.0**, Accelerate **1.12.0**, and bitsandbytes **0.50.2**. Both active models were loaded using 4-bit NF4 with double quantization, `device_map="auto"`, and BF16 compute when supported (otherwise FP16). The generator used `Qwen3_5ForConditionalGeneration` and `AutoProcessor`; the verifier used `AutoModelForCausalLM` and `AutoTokenizer`. This run did not use vLLM, Ollama, or a hosted inference API. The code set an **8,192-token operational context limit**, irrespective of a model card's larger native context length.

## 2. Generation approach and prompting

- The active method is **LLM-authored correct-sentence generation followed by deterministic reverse error injection**. For each attempt, the code selects one of six manually specified phrase pairs. Qwen3.5 writes a correct Vietnamese sentence containing the designated `fixed_span` exactly once. Python then replaces that span with the paired `error_span`, preserving all text outside the span. For example, `một chiếc bàn gỗ` becomes `một chiếc gỗ bàn`. The LLM did **not** freely write the erroneous sentence in the completed Modal run.
- The generator call is **zero-shot with structural constraints**: its active `controlled_seed_prompt` supplies a required phrase and variation value but no example input-output demonstrations. The verifier is **few-shot**: `classify_prompt` supplies eligible labeled user/assistant examples from the ten entries in `vgec/resources/fewshot.json`, after excluding examples that resemble the holdout or the current input. The verifier's final input omits the generator's target label, correction, subtype, and explanation. A second prompt audits the candidate and proposed correction as a pair.
- An alternate `injection_prompt` in the repository does request LLM-authored errors and includes label-2 examples, but **`controlled_injection=true` bypassed that prompt** in this run. The configured `hard_examples_path` is `null`, so hard-example mining did not supply demonstrations. No prompt asks for chain-of-thought reasoning; the verifier is asked for a short `reason` field, which should not be reported as a chain-of-thought trace.

**Representative active generator prompt template** (`vgec/prompts.py`, `controlled_seed_prompt`; Vietnamese text retained verbatim):

```text
System: Bạn viết câu tiếng Việt đúng ngữ pháp làm nền cho bài tập sửa lỗi. Chỉ trả một JSON object hợp lệ dạng {"sentence":"..."}, không markdown.

User: Viết một câu tự nhiên, hoàn chỉnh, khoảng 10–22 từ, với tình huống đơn giản phù hợp với cụm bắt buộc. Câu PHẢI chứa chính xác một lần cụm nguyên văn trong dữ liệu; giữ nguyên dấu và khoảng trắng của cụm. Không thêm mệnh đề giải thích hoặc cụm từ gượng ép chỉ để khớp chủ đề. Không đưa câu sai hoặc lời giải vào JSON.
DATA:
{"required_phrase":"<fixed_span>","variation":<seed>}
```

- **Generator decoding:** `temperature=0.8`; default `top_p=0.8`; `top_k=20`; `max_new_tokens=1024`; sampling enabled. The per-attempt random seed is `42 + 10 × attempt_index`. The prompt is rejected if prompt length plus the 1,024-token allowance exceeds 8,192 tokens.
- **Verifier decoding:** the default `temperature=0.0` is used for both blind classification and pair audit; sampling is disabled, so `top_p` and `top_k` are not passed to `generate()` for those calls. Its maximum is also 1,024 new tokens. The chat template is called with `enable_thinking=False`.
- These are **configured limits**, not observed token counts. The program does not record generated-token totals or aggregate token usage.

## 3. Source data and observed output

- No external seed-sentence file was used: `seed_path` defaults to `null`, and `configs/modal.json` does not override it. Qwen3.5 produced the correct source sentences from the required phrases in code.
- `VGEC 2026_v2.json` was uploaded to `/mnt/data/` and configured as a **holdout/exclusion corpus**, not as a source of generated seeds or corrections. Both its `Input` and available corrected references are excluded from synthetic candidates. The built-in few-shot resource is separate from this corpus.
- The completed run's `report.json` records **20 generation attempts**, **19 structurally valid candidates**, **1 generation rejection**, **19 verifier decisions**, and **19 accepted candidates**. The single generation rejection was `missing_or_repeated_required_phrase` at attempt index 17. All 19 accepted candidates were assigned blind verifier label 2; no verifier rejections were recorded.
- The export routine selected **10 of those 19 accepted candidates** and wrote **10 JSONL records**, IDs **120001–120010**, each with integer `Error_Type_Id: 2` and exactly one correction. The exported audit file contains five `noun_order` and five `verb_complement` candidates. The output count is an export target; it is not the total number of accepted candidates.
- The artifacts contain **no observed total token count**, so one should not derive such a figure from the 1,024-token per-call ceiling. There is also no recorded macro-F1 or official GEC development-set score for this generation run.

## 4. Post-processing and quality control

- **Model-output parsing:** `parse_object` requires one bounded JSON object; it accepts a single valid JSON code fence, rejects duplicate keys and nonfinite constants, and fails on malformed or extra JSON. The seed response must contain exactly the `sentence` key with valid text.
- **Deterministic edit validation:** the required `fixed_span` must occur exactly once in the generated correct sentence. `controlled_candidate` constructs the erroneous input by one span replacement. `validate_candidate` rejects unchanged inputs, repairs that do not reconstruct the exact correct sentence, and edits judged to involve only function words or punctuation. The latter is a conservative heuristic, not a proof of grammatical label.
- **Leakage and duplication controls:** Unicode NFC normalization, case folding, and whitespace normalization support exact and approximate exclusion. `Deduplicator` applies character-sequence similarity with a configured threshold of **0.9**. It checks generated sentences and erroneous inputs against the holdout, built-in examples, and previously retained candidates.
- **Independent label gate:** the verifier receives the erroneous input without the generator's proposed answer and must return integer label **2** with self-reported confidence **≥0.85**. A separate pair audit must affirm that the proposed correction is grammatical, meaning preserving, minimal, and confined to one phrase; it must reject repairs achievable solely through function words or punctuation and require `error_count=1`. The reported confidence is **not calibrated probability**, and acceptance is not human gold annotation.
- **Final schema gate:** export verifies the exact record structure, consecutive IDs, integer label 2, one nonempty correction, and uniqueness of both erroneous inputs and corrections. The locally downloaded dataset passed `python -m vgec validate ... --count 10`.

## Reporting qualification

The most faithful description of this run is **“zero-shot constrained Qwen3.5 sentence generation, deterministic phrase-level reverse error injection, and few-shot independent Qwen3 verification.”** Describing the active generator as few-shot error generation, claiming that Qwen2.5-7B was used, reporting 19 final records, or citing unmeasured token totals or dev-set scores would conflict with the code or run artifacts. The ten exported records are pipeline-approved synthetic examples; their linguistic validity has not been established by an independent human annotation study.
