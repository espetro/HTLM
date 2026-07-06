# Browser-Agent Schemas

JSON Schema definitions for the on-device browser-agent LLM.

The fine-tuned model is a **perception + grounding** API, not a planner. It receives a compact page representation and a natural-language instruction, and emits a single deterministic action.

## Files

- `page-representation.json` — schema for the serialized page.
- `action.json` — schema for the resolved action.
- `example.json` — one valid input/output pair.

## Page representation

`page-representation.json` describes a pruned, indexed list of interactive elements.

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `url` | string | no | Included when global context helps; omitted to save tokens when only local state matters. |
| `title` | string | no | Page title or accessible document name. |
| `elements` | array | yes | Ordered top-to-bottom, left-to-right. |
| `elements[].index` | integer | yes | 0-based, unique within the page. Used by the action schema to refer to the element. |
| `elements[].role` | string enum | yes | Small closed set: `button`, `link`, `input`, `textarea`, `select`, `checkbox`, `radio`, `combobox`, `searchbox`, `menu`, `menuitem`, `tab`, `switch`. Keeps the output distribution tight. |
| `elements[].label` | string | yes | Accessible name, `aria-label`, or visible text. Empty string allowed. |
| `elements[].placeholder` | string | no | Hint text for text-like controls. |
| `elements[].value` | string | no | Current value or state. |
| `elements[].tag` | string | no | Original HTML tag for DOM round-trip. |
| `elements[].bounds` | object | no | Optional `{x, y, width, height}` for future visual grounding. |
| `metadata.source` | string | no | Pipeline/extension that produced the representation. |
| `metadata.timestamp` | string (ISO 8601) | no | Extraction time. |

### Why this shape

- **Token efficiency:** only interactive, visible elements are included; static layout noise is stripped.
- **Model-friendly:** a flat array with small integer indices is easier to attend to and reproduce than raw HTML or XPath.
- **Deterministic parse:** each element has exactly one `index`, so the action schema can resolve targets without ambiguous selectors.

## Action

`action.json` describes the closed set of operations the executor can run.

| Variant | Fields | Meaning |
|---------|--------|---------|
| `click` | `index` | Click the element at `index`. |
| `type` | `index`, `text`, `submit` (default `false`) | Type `text` into the element at `index`; optionally press Enter. |
| `select` | `index`, `value` | Select option `value` on the element at `index`. |
| `scroll` | `direction` | Scroll the viewport `up`, `down`, `left`, or `right`. |
| `wait` | — | Wait for the page to settle. |
| `done` | `answer` (optional) | Task complete; optional answer string. |

The schema uses `oneOf` so each variant is validated independently and extra fields are rejected.

## Example

`example.json` shows a travel search page and the instruction "Search for a cheap flight from Madrid to Tokyo". The resolved action starts by typing "Madrid" into the origin combobox.
