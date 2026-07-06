"""Per-model chat template formatting for training.

Each model family has a different chat template convention.  This module provides
a formatter factory so `train.py` is model-agnostic.

Usage:
    from training.formatters import get_formatter
    formatter = get_formatter("qwen2.5-0.5b")
    messages = formatter.format(record)   # → list of message dicts
    text = formatter.apply_template(messages, tokenizer)  # → tokenized string
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

SYSTEM_PROMPT_GROUNDING = """\
You are a browser grounding model. You receive a JSON description of the \
interactive elements on a web page and a natural-language instruction. Output \
exactly ONE action as compact JSON.

Roles: button, link, input, textarea, select, checkbox, radio, combobox, \
searchbox, menu, menuitem, tab, switch. Elements have unique indices.

Actions:
- {"type":"click","index":<int>}
- {"type":"type","index":<int>,"text":"...","submit":false}
- {"type":"select","index":<int>,"value":"..."}
- {"type":"scroll","direction":"up"|"down"|"left"|"right"}
- {"type":"wait"}
- {"type":"done","answer":"..."}

Rules: index must reference an existing element. Prefer fewest steps. \
Output only the JSON object."""


class Formatter(ABC):
    """Base formatter interface."""

    name: str = ""

    @abstractmethod
    def format(self, record: dict[str, Any]) -> list[dict[str, str]]:
        """Return list of {role, content} messages for this record."""
        ...

    def apply_template(self, messages: list[dict[str, str]], tokenizer: Any) -> str:
        """Apply tokenizer's chat template. Returns raw string (not tokenized)."""
        # Most formatters use the standard HuggingFace chat template
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


class LlamaFormatter(Formatter):
    """Llama 3 / LFM2.5 / Mistral family — uses <|begin_of_text|>...<|end_of_text|>."""

    name = "llama"

    def format(self, record: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT_GROUNDING},
            {"role": "user", "content": self._render_page(record)},
            {"role": "assistant", "content": _compact_action(record["action"])},
        ]

    def _render_page(self, record: dict[str, Any]) -> str:
        import json

        page = dict(record.get("page") or {})
        page.pop("metadata", None)
        return f"{record['instruction']}\n\n{json.dumps(page, ensure_ascii=False, separators=(',', ':'))}"


class QwenFormatter(Formatter):
    """Qwen2.5 family — uses <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n..."""

    name = "qwen2.5"

    def format(self, record: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "user", "content": self._render_page(record)},
            {"role": "assistant", "content": _compact_action(record["action"])},
        ]
        # Note: Qwen's system prompt is set via generation_config.system_prefix
        # or passed to apply_chat_template as the "system" role message.

    def apply_template(self, messages: list[dict[str, str]], tokenizer: Any) -> str:
        # Qwen apply_chat_template handles the system message differently:
        # pass it as part of messages with role=system.
        system_msg = {"role": "system", "content": SYSTEM_PROMPT_GROUNDING}
        return tokenizer.apply_chat_template(
            [system_msg] + messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def _render_page(self, record: dict[str, Any]) -> str:
        import json

        page = dict(record.get("page") or {})
        page.pop("metadata", None)
        return f"{record['instruction']}\n\n{json.dumps(page, ensure_ascii=False, separators=(',', ':'))}"


class GemmaFormatter(Formatter):
    """Gemma 3 / FunctionGemma family — uses <|begin_of_turn|><|role|>...<|end_of_turn|>."""

    name = "gemma3"

    def format(self, record: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "user", "content": self._render_page(record)},
            {"role": "assistant", "content": _compact_action(record["action"])},
        ]

    def apply_template(self, messages: list[dict[str, str]], tokenizer: Any) -> str:
        # Gemma's tokenizer.apply_chat_template accepts the same interface as Llama
        system_msg = {"role": "system", "content": SYSTEM_PROMPT_GROUNDING}
        return tokenizer.apply_chat_template(
            [system_msg] + messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    def _render_page(self, record: dict[str, Any]) -> str:
        import json

        page = dict(record.get("page") or {})
        page.pop("metadata", None)
        return f"{record['instruction']}\n\n{json.dumps(page, ensure_ascii=False, separators=(',', ':'))}"


def _compact_action(action: dict[str, Any]) -> str:
    import json

    return json.dumps(action, ensure_ascii=False, separators=(",", ":"))


# ── factory ────────────────────────────────────────────────────────────────────

# Model → formatter class mapping.
# Uses substring matching so partial model names work.
_FORMATTERS: list[tuple[list[str], type[Formatter]]] = [
    (["qwen2.5", "qwen"], QwenFormatter),
    (["gemma", "functiongemma"], GemmaFormatter),
    (["lfm", "llama", "mistral", "liquid"], LlamaFormatter),
]


def get_formatter(model_id: str) -> Formatter:
    """Return the appropriate formatter for a model ID."""
    lower = model_id.lower()
    for aliases, cls in _FORMATTERS:
        if any(alias in lower for alias in aliases):
            return cls()
    # Fallback to Llama-style
    return LlamaFormatter()


def guess_formatter_name(model_id: str) -> str:
    """Return the formatter key for a model (used to match config.formatter)."""
    return get_formatter(model_id).name
