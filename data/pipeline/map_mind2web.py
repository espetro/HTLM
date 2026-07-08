"""Map Mind2Web (osunlp/Mind2Web or oottyy/Mind2Web_AXT) rows to HTLM training records.

Mind2Web rows look like:
  {
    "annotation_id": "abc123",
    "confirmed_task": "Find a mini van at Brooklyn...",
    "actions": [{
      "operation": {"op": "CLICK"|"TYPE"|"SELECT", "value": "..."},
      "pos_candidates": [{tag, is_original_target, backend_node_id, attributes}, ...],
      "neg_candidates": [...],
      "axtree_json": [{node_id, role, name, bounding_box_rect: "x,y,w,h"}, ...],  # AXTree variant only
    }, ...]
  }

For the bounding-box field we prefer oottyy/Mind2Web_AXT's axtree_json (absolute CSS px).
If that field is absent we fall back to no bounds (fine for training the model — bounds
are optional in page-representation.json).
"""

from __future__ import annotations

import json
from typing import Any

from data.pipeline.records import TrainingRecord


# ── role mapping ──────────────────────────────────────────────────────────────

def _map_role(tag: str, attributes_str: str | None = None) -> str | None:
    """Map an HTML tag (+ optional type attr) to our closed interactive role enum.

    Returns None for structural tags (div/span/p/li/...) — page-representation is
    the action surface, not the full DOM. Filtering them keeps element lists small
    enough for a ≤500M model and matches the schema's closed-enum intent.
    """
    t = tag.lower()
    if t == "input":
        if attributes_str:
            try:
                attrs = json.loads(attributes_str)
                it = attrs.get("type", "text").lower()
                if it in ("checkbox", "radio"):
                    return it
            except Exception:
                pass
        return "input"
    if t == "a":
        return "link"
    role_map = {
        "button": "button",
        "textarea": "textarea",
        "select": "select",
        "menu": "menu",
        "menuitem": "menuitem",
        "tab": "tab",
        "switch": "switch",
    }
    return role_map.get(t)


def _get_label(attributes_str: str | None, text: str | None = None) -> str:
    """Return the best label: aria-label > text > ''."""
    if attributes_str:
        try:
            attrs = json.loads(attributes_str)
            aria = attrs.get("aria-label")
            if aria and aria.strip():
                return aria.strip()
        except Exception:
            pass
    if text and text.strip():
        return text.strip()
    return ""


def _parse_bbox(bbox_str: str | None) -> dict[str, float] | None:
    """Parse 'x,y,w,h' in CSS pixels into bounds dict, or None."""
    if not bbox_str:
        return None
    parts = bbox_str.split(",")
    if len(parts) != 4:
        return None
    try:
        x, y, w, h = map(float, parts)
        return {"x": x, "y": y, "width": w, "height": h}
    except Exception:
        return None


MAX_ELEMENTS = 64
# ponytail: cap element list so it fits max_seq_length≈2048 with room for instruction+action.
# pos candidates (the target's neighborhood) go first so the target always survives the cap;
# neg candidates fill the rest as distractors. A consistent pos-first ordering is fair because
# both train and eval sets come from the same mapper. Caveat: biases target toward low indices;
# if generalization to high-index targets matters, interleave deterministically.


def _build_elements(pos: list[dict], neg: list[dict], axtree: list[dict] | None) -> list[dict]:
    """Merge pos+neg candidates into a flat indexed elements list (interactable only).

    Only elements whose tag maps to a valid role are kept (structural tags dropped).
    pos first, then neg, capped at MAX_ELEMENTS so the list fits a small model's context.
    """
    merged: list[dict] = []
    seen_backend: set[str] = set()
    idx = 0
    for bucket in [pos, neg]:
        for el in bucket:
            if len(merged) >= MAX_ELEMENTS:
                break
            bid = str(el.get("backend_node_id") or "")
            if bid and bid in seen_backend:
                continue
            if bid:
                seen_backend.add(bid)
            tag = str(el.get("tag") or "")
            attrs_str = el.get("attributes")
            role = _map_role(tag, attrs_str)
            if role is None:
                continue
            label = _get_label(attrs_str, el.get("text"))
            element: dict[str, Any] = {"index": idx, "role": role, "label": label, "tag": tag}
            # bounds: try axtree first, then skip
            if axtree:
                bid_key = el.get("backend_node_id")
                for node in axtree:
                    if str(node.get("backend_id") or "") == str(bid_key):
                        element["bounds"] = _parse_bbox(node.get("bounding_box_rect"))
                        break
            merged.append(element)
            idx += 1
    return merged


def _find_target_index(elements: list[dict], pos: list[dict], neg: list[dict]) -> int | None:
    """Find the merged-list index of the canonical target element."""
    all_candidates = neg + pos
    for cand in all_candidates:
        if cand.get("is_original_target"):
            # match by tag + label heuristic (backend_node_id may be empty)
            tag = str(cand.get("tag") or "")
            label = _get_label(cand.get("attributes"), cand.get("text"))
            for el in elements:
                if el.get("tag") == tag and el.get("label") == label:
                    return el["index"]
    return None


def map_row(row: dict) -> list[TrainingRecord]:
    """Convert one Mind2Web row (one task) into a list of TrainingRecords (one per step)."""
    task_id = row.get("annotation_id") or row.get("task_id", "")
    instruction = row.get("confirmed_task") or row.get("task") or ""

    records = []
    for step in row.get("actions") or []:
        op = step.get("operation") or {}
        op_type = op.get("op", "").upper()
        op_value = op.get("value") or ""

        pos: list[dict] = step.get("pos_candidates") or []
        neg: list[dict] = step.get("neg_candidates") or []
        axtree = step.get("axtree_json") if "axtree_json" in step else None

        elements = _build_elements(pos, neg, axtree)
        target_idx = _find_target_index(elements, pos, neg)
        if target_idx is None:
            # target was structural-tagged (filtered out) or unfound → skip rather
            # than mislabel as index 0.
            continue

        action: dict[str, Any] | None = None
        if op_type == "CLICK":
            action = {"type": "click", "index": target_idx}
        elif op_type == "TYPE":
            action = {"type": "type", "index": target_idx, "text": str(op_value), "submit": False}
        elif op_type == "SELECT":
            action = {"type": "select", "index": target_idx, "value": str(op_value)}
        # HOVER, ENTER, etc. → skip (no mapping)

        if action is None:
            continue

        page: dict[str, Any] = {"elements": elements}
        records.append(
            TrainingRecord(
                instruction=instruction,
                page=page,
                action=action,
                meta={"task_id": task_id, "source": "mind2web"},
            )
        )
    return records
