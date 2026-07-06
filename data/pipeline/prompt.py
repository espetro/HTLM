"""Fine-tuning prompt template for the grounding model.

The template is model-agnostic: it produces {system, user, assistant} message
dicts. The training script applies the base model's chat template
(`tokenizer.apply_chat_template`), so one template serves Qwen / Gemma / LFM2.

This is the core IP of the perception+grounding contract. Keep it stable once
a bake-off is locked in, otherwise eval numbers stop being comparable.
"""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """\
You are a browser grounding model. You receive a JSON description of the \
interactive elements currently on a web page plus a natural-language \
instruction. You choose exactly ONE action that makes progress toward the \
instruction and reply with ONLY that action as compact JSON.

The page JSON looks like {"elements":[{"index":0,"role":"button","label":\
"Search", ...}, ...]}. Every element has a unique "index". Roles are one of: \
button, link, input, textarea, select, checkbox, radio, combobox, searchbox, \
menu, menuitem, tab, switch.

Reply with exactly one of these objects and nothing else (no markdown, no prose):
- {"type":"click","index":<int>}                      click the element at index
- {"type":"type","index":<int>,"text":"...","submit":false}   fill a field; submit true presses Enter
- {"type":"select","index":<int>,"value":"..."}       choose an option in a select/combobox
- {"type":"scroll","direction":"up"|"down"|"left"|"right"}
- {"type":"wait"}                                      page still loading, no productive action yet
- {"type":"done","answer":"..."}                       instruction fully satisfied

Rules:
- For click/type/select, "index" must reference an element index that exists on the page.
- Prefer the fewest steps: if one type+submit finishes a search, emit it as a single action.
- Output only the JSON object.
"""


def _compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def render_page(page: dict[str, Any]) -> str:
    """Compact page JSON with metadata stripped (carries no grounding signal)."""
    page = dict(page or {})
    page.pop("metadata", None)
    return _compact(page)


def input_messages(record: dict[str, Any], system: str | None = None) -> list[dict[str, str]]:
    """system + user only. Used for inference and teacher distillation."""
    return [
        {"role": "system", "content": system or SYSTEM_PROMPT},
        {"role": "user", "content": f"{record['instruction']}\n\n{render_page(record.get('page') or {})}"},
    ]


def training_messages(record: dict[str, Any], system: str | None = None) -> list[dict[str, str]]:
    """system + user + assistant. Used to build fine-tuning examples."""
    return input_messages(record, system) + [
        {"role": "assistant", "content": _compact(record["action"])}
    ]
