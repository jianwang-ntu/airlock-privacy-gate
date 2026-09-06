"""Where a document goes after the gate has ruled on it.

Two destinations:

  hosted   the model the operator wants to use and Airlock is protecting them
           from. THIS REPOSITORY CALLS NO THIRD-PARTY API. `HostedStandIn` runs
           a larger model on the same machine so the whole loop -- redact, ask,
           re-hydrate -- can be exercised and measured end to end without any
           credential and without any document leaving the host. Point
           `HostedAdapter` at a real provider to use one; that leg has not been
           exercised here and nothing in the evidence files describes it.

  local    the fallback for documents the gate refuses to forward. A small
           model on the operator's own machine. The answer is worse; the
           document stays home. That trade is the product.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

LOCAL_DEFAULT = "Qwen/Qwen2.5-0.5B-Instruct"
HOSTED_STANDIN_DEFAULT = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass
class Answer:
    text: str
    model: str
    seconds: float
    route: str


class LocalLM:
    """A causal LM answering on this machine. Loaded lazily."""

    def __init__(self, model_id: str = LOCAL_DEFAULT, route: str = "local_model",
                 device: str | None = None, max_new_tokens: int = 256):
        self.model_id = model_id
        self.route = route
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._tok = self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()

    def answer(self, prompt: str, system: str | None = None) -> Answer:
        self._load()
        t0 = time.time()
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        enc = self._tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        import torch

        with torch.no_grad():
            out = self._model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
        text = self._tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return Answer(text=text.strip(), model=self.model_id,
                      seconds=round(time.time() - t0, 2), route=self.route)


class HostedStandIn(LocalLM):
    """A local stand-in for the hosted model. Nothing leaves the machine."""

    def __init__(self, model_id: str = HOSTED_STANDIN_DEFAULT, **kw):
        super().__init__(model_id=model_id, route="hosted_model_standin", **kw)


class HostedAdapter:
    """Adapter for a real hosted provider. Not exercised in this repository.

    `call` is any callable taking (prompt, system) and returning a string. No
    provider client, endpoint or credential ships here, and no measurement in
    `evidence/` was produced through this class.
    """

    def __init__(self, call, name: str = "hosted_provider"):
        self.call = call
        self.model_id = name
        self.route = "hosted_model"

    def answer(self, prompt: str, system: str | None = None) -> Answer:
        t0 = time.time()
        text = self.call(prompt, system)
        return Answer(text=text, model=self.model_id,
                      seconds=round(time.time() - t0, 2), route=self.route)


# The template pattern is deliberately NOT spelled out here. An earlier version
# said "tokens of the form [TYPE_n]" and Qwen3-4B echoed that literal back in a
# footnote, which the integrity check then reported as two placeholders the
# vault had never issued. Describing the shape in words costs nothing.
PRESERVE_SYSTEM = (
    "You are answering about a redacted document. Some identifiers have been "
    "replaced by short placeholders written in square brackets. Copy any "
    "placeholder you refer to exactly as it appears in the document, brackets "
    "included. Never invent a value in place of one, and never introduce a "
    "bracketed placeholder that is not already in the document."
)
