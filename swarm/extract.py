"""Policy extraction: P(the first answer token is the A label | history prompt), constrained to the two labels.

For each history, framing and label mapping we record the probability mass on tokens that equal "J" or "K"
after stripping whitespace (so 'J' and ' J' are summed), plus how much mass went elsewhere. The policy is
then p_T(A) = sigmoid(log(mass_A / mass_B) / T). This is the renormalized two-label policy, not the
unrestricted full-text policy.

Backends:
  adaption    Adaption hosted model (Responses API). No logprobs, so P(label) is estimated by sampling n times.
  openrouter  hosted model with top_logprobs. A label missing from the truncated list is NOT zero; it is
              flagged and bounded above by the smallest listed probability.
  local       Hugging Face model on this machine (MPS/CPU). Exact probabilities, optional LoRA adapter.
              Needs: pip install torch transformers peft (and model weights downloaded from Hugging Face).
"""

import json
import math
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .game import A, B, HistorySpace
from .prompts import LABEL_MAPPINGS, build_prompt, prompt_set_hash

LABELS = ("J", "K")


def label_masses(top: list[tuple[str, float]]) -> dict:
    """Sum probability over listed tokens equal to each label after stripping whitespace."""
    mass = {L: 0.0 for L in LABELS}
    found = {L: False for L in LABELS}
    listed = 0.0
    for tok, lp in top:
        pr = math.exp(lp)
        listed += pr
        s = tok.strip()
        if s in mass:
            mass[s] += pr
            found[s] = True
    min_listed = min((math.exp(lp) for _, lp in top), default=0.0)
    return {"mass": mass, "found": found, "listed_mass": listed, "min_listed": min_listed}


class OpenRouterBackend:
    name = "openrouter"

    def __init__(self, model: str, provider: str | None = None, top_logprobs: int = 20, disable_reasoning=True):
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(override=True)
        self.client = OpenAI(
            base_url=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=os.environ["OPENROUTER_API_KEY"],
            max_retries=4,
        )
        self.model, self.provider, self.top_logprobs = model, provider, top_logprobs
        self.extra = {"usage": {"include": True}}
        self.extra["provider"] = (
            {"order": [provider], "allow_fallbacks": False, "require_parameters": True}
            if provider
            else {"require_parameters": True}
        )
        if disable_reasoning:
            self.extra["reasoning"] = {"enabled": False}
        self.meta = {"backend": self.name, "model": model, "provider_pin": provider,
                     "top_logprobs": top_logprobs, "disable_reasoning": disable_reasoning}

    def score(self, prompt: str) -> dict:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=1.0,  # temperature is applied afterwards to the label logits, never here
            logprobs=True,
            top_logprobs=self.top_logprobs,
            extra_body=self.extra,
        )
        content = r.choices[0].logprobs.content if r.choices[0].logprobs else None
        if not content:
            raise RuntimeError(f"no logprobs returned (provider={getattr(r, 'provider', '?')})")
        top = [(t.token, t.logprob) for t in content[0].top_logprobs]
        out = label_masses(top)
        usage = r.usage.model_dump() if r.usage else {}
        out.update(
            top=top[:5],
            provider=getattr(r, "provider", None),
            served_model=r.model,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cost=usage.get("cost"),
        )
        return out


class RateLimiter:
    """Process-wide adaptive limiter for Adaption's per-user inference rate limit (HTTP 429, retry-after 60 s).

    Spaces calls at `rate` per second across all threads. On a 429 every thread pauses for retry-after and the
    rate is cut by 30% (once per pause window); each success nudges the rate back up toward max_rate.
    """

    def __init__(self, rate: float = 8.0, max_rate: float = 12.0, min_rate: float = 1.0):
        self.rate, self.max_rate, self.min_rate = rate, max_rate, min_rate
        self.lock = threading.Lock()
        self.next_slot = time.monotonic()
        self.paused_until = 0.0
        self.n_429 = 0
        self.n_ok = 0

    def acquire(self):
        with self.lock:
            start = max(time.monotonic(), self.next_slot, self.paused_until)
            self.next_slot = start + 1.0 / self.rate
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)

    def on_429(self, retry_after: float):
        with self.lock:
            self.n_429 += 1
            now = time.monotonic()
            if now >= self.paused_until:  # first 429 of this window: cut the rate once
                self.rate = max(self.min_rate, self.rate * 0.7)
            self.paused_until = max(self.paused_until, now + retry_after)
            self.next_slot = max(self.next_slot, self.paused_until)

    def on_ok(self):
        with self.lock:
            self.n_ok += 1
            self.rate = min(self.max_rate, self.rate + 0.01)

    def status(self) -> str:
        return f"rate {self.rate:.1f}/s, ok {self.n_ok}, 429s {self.n_429}"


_max_rate = float(os.environ.get("ADAPTION_MAX_RATE", "12"))
ADAPTION_LIMITER = RateLimiter(rate=min(8.0, _max_rate), max_rate=_max_rate)


def clean_answer(text: str) -> str | None:
    """'J', ' J.', '**J**' -> 'J'. Anything that is not a bare label is invalid (None), and is counted."""
    s = (text or "").strip().strip("*`'\".:!() \n")
    return s if s in LABELS else None


class AdaptionSampleBackend:
    """Adaption hosted model via the Responses API.

    Adaption does not return token probabilities, so P(label) is estimated from n independent, stateless
    samples (store=False, agentic_memory=False, no stack). Masses use a Jeffreys prior, (k + 0.5) / (n + 1),
    so no estimate is exactly 0 or 1. Invalid answers are kept as a measured outcome.
    """

    name = "adaption"

    def __init__(self, model: str, samples: int = 200, temperature: float = 1.0, reasoning_effort="none",
                 inner_workers: int = 8, max_output_tokens: int = 8):
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(override=True)
        self.client = OpenAI(
            base_url=os.environ.get("ADAPTION_BASE_URL", "https://api.prod.adaptionlabs.ai/api/v1"),
            api_key=os.environ["ADAPTION_API_KEY"],
            max_retries=0,  # retries and 429 back-off are handled in _one with the shared limiter
            timeout=30,  # the client default is 10 minutes, which lets a few hung calls stall a whole run
        )
        self.limiter = ADAPTION_LIMITER
        self.model, self.samples, self.temperature = model, samples, temperature
        self.inner_workers, self.max_output_tokens = inner_workers, max_output_tokens
        self.extra = {"agentic_memory": False}
        if reasoning_effort:
            self.extra["reasoning"] = {"effort": reasoning_effort}
        # Adaption applies top_p = 0.95 unless told otherwise (seen in its response metadata), so labels under ~5%
        # are never sampled. ADAPTION_TOP_P overrides it for diagnostics; the confirmatory runs keep the default.
        tp = os.environ.get("ADAPTION_TOP_P")
        self.top_p = float(tp) if tp else None
        self.meta = {"backend": self.name, "model": model, "samples_per_prompt": samples,
                     "sample_temperature": temperature, "reasoning_effort": reasoning_effort,
                     "max_output_tokens": max_output_tokens, "estimator": "Jeffreys (k+0.5)/(n+1)",
                     "top_p": self.top_p if self.top_p is not None else "server default (0.95)"}

    def _one(self, prompt: str):
        from openai import RateLimitError

        failures, last = 0, ""
        for _ in range(30):  # 429s are waited out; other failures get 3 attempts
            self.limiter.acquire()
            try:
                r = self.client.responses.create(model=self.model, input=prompt,
                                                 max_output_tokens=self.max_output_tokens,
                                                 temperature=self.temperature, store=False, extra_body=self.extra,
                                                 **({"top_p": self.top_p} if self.top_p is not None else {}))
            except RateLimitError as e:
                ra = e.response.headers.get("retry-after") if getattr(e, "response", None) is not None else None
                self.limiter.on_429(float(ra) if ra else 60.0)
                continue
            except Exception as e:
                failures += 1
                last = repr(e)[:200]
                if failures >= 3:
                    break
                continue
            self.limiter.on_ok()
            u = r.usage
            return r.output_text, None, u.input_tokens, u.output_tokens, float(getattr(u, "cost", 0) or 0)
        return None, last or "gave up after repeated 429s", 0, 0, 0.0

    def score(self, prompt: str) -> dict:
        with ThreadPoolExecutor(max_workers=self.inner_workers) as ex:
            outs = list(ex.map(lambda _: self._one(prompt), range(self.samples)))
        errors = [o for o in outs if o[1]]
        answers = [clean_answer(o[0]) for o in outs if not o[1]]
        k = {L: answers.count(L) for L in LABELS}
        n = len(answers)
        if n == 0:
            raise RuntimeError(f"all {self.samples} samples failed, e.g. {errors[0][1]}")
        invalid = [o[0] for o in outs if not o[1] and clean_answer(o[0]) is None]
        return {
            "mass": {L: (k[L] + 0.5) / (n + 1) for L in LABELS},
            "found": {L: True for L in LABELS},
            "listed_mass": 1.0,
            "min_listed": 0.0,
            "counts": {**k, "invalid": len(invalid), "errors": len(errors), "n": n},
            "invalid_examples": invalid[:3],
            "top": [],
            "provider": "adaption",
            "served_model": self.model,
            "prompt_tokens": sum(o[2] for o in outs),
            "completion_tokens": sum(o[3] for o in outs),
            "cost": sum(o[4] for o in outs),
        }


def prepare_text_only_adapter(path: str) -> str:
    """AutoScientist trains Qwen3.5 as a vision-language model, so LoRA keys read 'model.language_model.layers…'.
    The text-only causal LM names them 'model.layers…'. Write a renamed copy next to the original if needed."""
    import shutil

    from safetensors.torch import load_file, save_file

    src = os.path.join(path, "adapter_model.safetensors")
    tensors = load_file(src)
    if not any(".language_model." in k for k in tensors):
        return path
    out = path.rstrip("/") + "_textonly"
    os.makedirs(out, exist_ok=True)
    save_file({k.replace("model.language_model.", "model."): v for k, v in tensors.items()},
              os.path.join(out, "adapter_model.safetensors"))
    shutil.copy(os.path.join(path, "adapter_config.json"), os.path.join(out, "adapter_config.json"))
    return out


class LocalHFBackend:
    name = "local"

    def __init__(self, model_id: str, adapter: str | None = None, device: str | None = None, dtype="bfloat16",
                 revision: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        # Text-only load: for vision-language checkpoints (e.g. Qwen3.5) the vision tower is skipped.
        model, info = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=getattr(torch, dtype),
                                                           output_loading_info=True)
        if info.get("missing_keys"):  # randomly initialized weights would give meaningless probabilities
            raise RuntimeError(f"{len(info['missing_keys'])} weights missing, e.g. {info['missing_keys'][:3]}")
        self.load_info = {k: len(v) for k, v in info.items() if hasattr(v, "__len__")}
        self.adapter_info = None
        if adapter:
            from peft import PeftModel

            adapter = prepare_text_only_adapter(adapter)
            model = PeftModel.from_pretrained(model, adapter)
            # PEFT initialises lora_B to zero, so any lora_B still all-zero means its weights did not load
            lora_b = [(n, p) for n, p in model.named_parameters() if "lora_B" in n]
            loaded = sum(bool(p.detach().abs().sum() > 0) for _, p in lora_b)
            if not lora_b or loaded < len(lora_b):
                raise RuntimeError(f"adapter {adapter}: only {loaded}/{len(lora_b)} LoRA B matrices loaded")
            self.adapter_info = {"lora_modules": len(lora_b), "loaded": loaded}
        self.model = model.to(self.device).eval()
        # every vocabulary id that decodes to exactly a label after stripping whitespace
        self.label_ids = {L: [] for L in LABELS}
        for i in range(len(self.tok)):
            s = self.tok.decode([i]).strip()
            if s in self.label_ids:
                self.label_ids[s].append(i)
        self.meta = {"backend": self.name, "model": model_id, "revision": revision, "adapter": adapter,
                     "adapter_info": self.adapter_info,
                     "device": self.device, "dtype": dtype, "label_token_ids": self.label_ids,
                     "loading_info": self.load_info, "torch": torch.__version__}

    def _chat_text(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        try:  # Qwen-style templates accept enable_thinking; others ignore or reject it
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def score(self, prompt: str) -> dict:
        torch = self.torch
        ids = self.tok(self._chat_text(prompt), return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**ids).logits[0, -1].float()
        probs = torch.softmax(logits, dim=-1)
        mass = {L: float(probs[idx].sum()) for L, idx in self.label_ids.items()}
        top_p, top_i = torch.topk(probs, 5)
        top = [(self.tok.decode([int(i)]), math.log(float(p))) for p, i in zip(top_p, top_i)]
        return {"mass": mass, "found": {L: True for L in LABELS}, "listed_mass": 1.0, "min_listed": 0.0,
                "top": top, "provider": "local", "served_model": self.meta["model"],
                "prompt_tokens": int(ids["input_ids"].shape[1]), "completion_tokens": 0, "cost": 0.0}


def extract(backend, H=2, framings=(0, 1), mappings=(0, 1), repeats=2, workers=8, max_calls=2000,
            checkpoint: str | None = None, game: str = "coord") -> dict:
    """Score every prompt. With a checkpoint path, finished prompts are appended as JSON lines as they complete,
    and a rerun resumes from them. Progress is printed every 5%."""
    space = HistorySpace(H)
    jobs = [(f, m, i, r) for f in framings for m in mappings for i in range(len(space)) for r in range(repeats)]
    n_calls = len(jobs) * getattr(backend, "samples", 1)
    if n_calls > max_calls:
        raise SystemExit(f"{n_calls} API calls exceeds --max-calls {max_calls}; raise it deliberately if intended")

    def one(job):
        f, m, i, r = job
        lab = LABEL_MAPPINGS[m]
        prompt = build_prompt(space.histories[i], f, m, game)
        for attempt in range(3):
            try:
                s = backend.score(prompt)
                break
            except Exception as e:  # transport failures only get retried; the error is logged
                err = repr(e)[:300]
                time.sleep(2 * (attempt + 1))
        else:
            return {"history": space.key(space.histories[i]), "framing": f, "mapping": m, "repeat": r, "error": err}
        return {
            "history": space.key(space.histories[i]),
            "framing": f,
            "mapping": m,
            "repeat": r,
            "label_A": lab[A],
            "mass_A": s["mass"][lab[A]],
            "mass_B": s["mass"][lab[B]],
            "found_A": s["found"][lab[A]],
            "found_B": s["found"][lab[B]],
            "listed_mass": s["listed_mass"],
            "min_listed": s["min_listed"],
            "top": s["top"],
            "provider": s.get("provider"),
            "served_model": s.get("served_model"),
            "prompt_tokens": s.get("prompt_tokens"),
            "completion_tokens": s.get("completion_tokens"),
            "cost": s.get("cost"),
            **({"counts": s["counts"], "invalid_examples": s["invalid_examples"]} if "counts" in s else {}),
        }

    records, done = [], set()
    if checkpoint and os.path.exists(checkpoint):
        for line in open(checkpoint):
            r = json.loads(line)
            records.append(r)
            done.add((r["framing"], r["mapping"], r["history"], r["repeat"]))
    todo = [j for j in jobs if (j[0], j[1], space.key(space.histories[j[2]]), j[3]) not in done]
    print(f"{len(records)} prompts resumed from checkpoint, {len(todo)} to go", flush=True)

    t0 = time.time()
    step = max(1, len(todo) // 20)
    ck = open(checkpoint, "a") if checkpoint else None
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(one, j) for j in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            records.append(r)
            if ck and "error" not in r:
                ck.write(json.dumps(r) + "\n")
                ck.flush()
            if k % step == 0 or k == len(todo):
                el = time.time() - t0
                lim = getattr(backend, "limiter", None)
                print(f"progress {k}/{len(todo)} prompts ({k / len(todo):.0%}), {el:.0f}s elapsed, "
                      f"~{el / k * (len(todo) - k):.0f}s left, errors {sum('error' in x for x in records)}"
                      + (f", {lim.status()}" if lim else ""), flush=True)
    if ck:
        ck.close()
    return {
        "meta": {
            **backend.meta,
            "H": H,
            "game": game,
            "framings": list(framings),
            "mappings": list(mappings),
            "repeats": repeats,
            "prompt_hash": prompt_set_hash(space.histories, framings, list(mappings), game),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "seconds": round(time.time() - t0, 1),
            "calls": n_calls,
            "errors": sum("error" in r for r in records),
            "total_cost": sum(r.get("cost") or 0 for r in records),
            "providers": sorted({r.get("provider") or "?" for r in records if "error" not in r}),
        },
        "records": records,
    }


def load_policy(path: str, framing: int, mapping: int, temperature: float = 1.0):
    """Return (pA over history codes, HistorySpace, diagnostics) for one framing x mapping at temperature T."""
    data = json.load(open(path))
    meta = data["meta"]
    space = HistorySpace(meta["H"])
    by_hist = defaultdict(list)
    for r in data["records"]:
        if r["framing"] == framing and r["mapping"] == mapping and "error" not in r:
            by_hist[r["history"]].append(r)

    pA = np.empty(len(space))
    spread, valid, bounded = [], [], []
    n_samples = n_invalid = 0
    for code, h in enumerate(space.histories):
        key = space.key(h)
        rs = by_hist.get(key)
        if not rs:
            raise ValueError(f"no records for history {key} (framing {framing}, mapping {mapping}) in {path}")
        mA, mB, per_rep = [], [], []
        for r in rs:
            a = r["mass_A"] if r["found_A"] else r["min_listed"]  # upper bound, flagged below
            b = r["mass_B"] if r["found_B"] else r["min_listed"]
            if not (r["found_A"] and r["found_B"]):
                bounded.append(key)
            mA.append(a)
            mB.append(b)
            per_rep.append(a / (a + b) if a + b > 0 else 0.5)
            valid.append(r["mass_A"] + r["mass_B"])
            if "counts" in r:
                n_samples += r["counts"]["n"]
                n_invalid += r["counts"]["invalid"]
        a, b = float(np.mean(mA)), float(np.mean(mB))
        logit = math.log(max(a, 1e-300)) - math.log(max(b, 1e-300))
        pA[code] = 1 / (1 + math.exp(-logit / temperature))
        spread.append(max(per_rep) - min(per_rep))

    diag = {
        "meta": meta,
        "framing": framing,
        "mapping": mapping,
        "temperature": temperature,
        "max_repeat_spread": float(max(spread)),
        "mean_valid_mass": float(np.mean(valid)),
        "min_valid_mass": float(np.min(valid)),
        "bounded_histories": sorted(set(bounded)),
        "samples": n_samples,
        "invalid_rate": (n_invalid / n_samples) if n_samples else None,
    }
    return pA, space, diag
