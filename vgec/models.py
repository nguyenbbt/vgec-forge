"""Small model boundary; API mode uses only the Python standard library."""

import gc
import json
import os
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ModelError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward an Authorization header to another host.


urlopen = build_opener(NoRedirect).open


def parse_object(content):
    if not isinstance(content, str) or len(content) > 1_000_000:
        raise ValueError("Expected bounded text model output")
    content = content.strip()
    if content.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
        if not match:
            raise ValueError("Invalid JSON fence")
        content = match.group(1)

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    result = json.loads(content, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


class APIBackend:
    def __init__(self, config):
        self.config = config
        self.model = config["model"]
        self.base = config["base_url"].rstrip("/")
        parts = urlsplit(self.base)
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("API URL must not contain credentials, query or fragment")
        if parts.scheme != "https" and not (parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1")):
            raise ValueError("Use HTTPS for remote APIs; HTTP is allowed on loopback only")

    def complete(self, messages, temperature=0.0, seed=42):
        payload = {"model": self.model, "messages": messages, "temperature": temperature,
                   "max_tokens": self.config.get("max_tokens", 2048), "seed": seed,
                   "top_p": self.config.get("top_p", 0.8)}
        extra = self.config.get("extra_body", {})
        if set(extra) & set(payload):
            raise ValueError("extra_body cannot override core request fields")
        payload.update(extra)
        headers = {"Content-Type": "application/json"}
        key_env = self.config.get("api_key_env")
        if key_env:
            key = os.environ.get(key_env)
            if not key:
                raise ModelError(f"Missing environment variable: {key_env}")
            headers["Authorization"] = "Bearer " + key
        request = Request(self.base + "/chat/completions", json.dumps(payload).encode(), headers)
        retries = self.config.get("retries", 2)
        for attempt in range(retries + 1):
            try:
                with urlopen(request, timeout=self.config.get("timeout", 180)) as response:
                    raw = response.read(2_000_001)
                if len(raw) > 2_000_000:
                    raise ModelError("API response exceeds size limit")
                body = json.loads(raw)
                choice = body["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ModelError("Model response incomplete; increase max_tokens or disable thinking")
                return parse_object(choice["message"]["content"])
            except HTTPError as exc:
                if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == retries:
                    raise ModelError(f"API HTTP {exc.code} for model {self.model}") from None
            except (URLError, TimeoutError, ConnectionError):
                if attempt == retries:
                    raise ModelError(f"API connection failed for model {self.model}") from None
            except (KeyError, IndexError, TypeError) as exc:
                raise ModelError("API response does not match chat completion format") from exc
            time.sleep(min(2 ** attempt, 8))
        raise ModelError("API retries exhausted")

    def close(self):
        pass


class TransformersBackend:
    def __init__(self, config):
        self.config = config
        self.model = config["model"]
        self.loaded = None
        self.processor = None

    def _load(self):
        if self.loaded is not None:
            return
        try:
            import torch
            import transformers as hf
        except ImportError as exc:
            raise ModelError('Install local extras with: python -m pip install -e ".[local]"') from exc
        quantized = self.config.get("load_in_4bit", True)
        if quantized and not torch.cuda.is_available():
            raise ModelError("4-bit local profile requires CUDA; select API or disable load_in_4bit")
        kwargs = {"device_map": self.config.get("device_map", "auto"),
                  "dtype": "auto", "trust_remote_code": False,
                  "revision": self.config.get("revision", "main")}
        if quantized:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            kwargs["quantization_config"] = hf.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        # Qwen3.5 full checkpoints include vision configuration even for text input.
        # https://huggingface.co/docs/transformers/model_doc/qwen3_5
        self.multimodal = "Qwen3.5" in self.model
        model_class = hf.Qwen3_5ForConditionalGeneration if self.multimodal else hf.AutoModelForCausalLM
        tokenizer_class = hf.AutoProcessor if self.multimodal else hf.AutoTokenizer
        self.processor = tokenizer_class.from_pretrained(self.model, trust_remote_code=False,
                                                         revision=kwargs["revision"])
        self.loaded = model_class.from_pretrained(self.model, **kwargs).eval()

    def complete(self, messages, temperature=0.0, seed=42):
        self._load()
        import torch
        from transformers import set_seed

        set_seed(seed)
        if self.multimodal:
            messages = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                        for m in messages]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True,
            return_tensors="pt", enable_thinking=False).to(self.loaded.device)
        prompt_length = inputs["input_ids"].shape[-1]
        if prompt_length + self.config.get("max_tokens", 2048) > self.config.get("context_limit", 8192):
            raise ModelError("Prompt exceeds configured local context limit; no silent truncation")
        kwargs = {"max_new_tokens": self.config.get("max_tokens", 2048), "do_sample": temperature > 0}
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=self.config.get("top_p", 0.8), top_k=20)
        with torch.inference_mode():
            output = self.loaded.generate(**inputs, **kwargs)
        tokens = output[0, prompt_length:]
        eos = self.loaded.generation_config.eos_token_id
        endings = eos if isinstance(eos, list) else [eos]
        if tokens.numel() and int(tokens[-1]) not in endings:
            raise ModelError("Local generation did not finish with EOS; increase max_tokens")
        text = self.processor.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return parse_object(text)

    def close(self):
        self.loaded = None
        self.processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def make_backend(config):
    kind = config.get("backend", "api")
    if kind == "api":
        return APIBackend(config)
    if kind == "transformers":
        return TransformersBackend(config)
    raise ValueError(f"Unsupported backend: {kind}")
