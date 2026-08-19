# DisputeDesk

A payment-dispute / chargeback copilot. It classifies a customer’s complaint into a Stripe/Razorpay-style reason code, then (separately) recommends contest / accept / review from synthetic order records.

**Current milestone: v2.** A FastAPI endpoint sends the customer message to Claude, then Pydantic validates the JSON into a structured classification.

## What it does

Two pieces sit side by side:

1. **Classify** (v2) — free-text customer message → Claude → `DraftClassification` (`reason`, `confidence`, `explanation`).
2. **Recommend** (v1) — a known `Dispute` + `Transaction` → a rules-based action string.

Reason codes:

| Reason | Meaning |
|---|---|
| `unrecognized` | Customer does not recognize the charge |
| `product_not_received` | Paid, but nothing arrived |
| `duplicate` | Charged more than once |
| `product_unacceptable` | Received, but wrong / damaged / not as described |

Action rules (`recommend_action`):

| Reason | Evidence | Action |
|---|---|---|
| `product_not_received` | Delivered **and** address matches billing | contest |
| `product_not_received` | No solid delivery proof | accept |
| `duplicate` | — | contest (look for a second matching transaction) |
| `unrecognized` | Delivery evidence exists | contest |
| `unrecognized` | No delivery evidence | needs_review |
| `product_unacceptable` | — | needs_review (not handled yet) |

Records live in in-memory lists in `disputedesk.py` (T1–T4 / D1–D4), standing in for a real orders database.

## Setup

Python **3.11+**. The repo pins `3.11.9` in `.python-version` so pyenv does not pick up a global Python 2 install.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install fastapi uvicorn pydantic anthropic python-dotenv
```

Create a `.env` in the project root (do not commit it):

```
ANTHROPIC_API_KEY=your_key_here
```

## Run

**CLI classifier** — classifies a sample customer message and prints the structured result:

```bash
python disputedesk.py
```

**API** — FastAPI app, reload on save:

```bash
uvicorn disputedesk:app --reload
```

Then:

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"customer_message": "The product I received is not the one I ordered."}'
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

`POST /classify` body:

```json
{ "customer_message": "I never placed this order." }
```

Response:

```json
{
  "reason": "unrecognized",
  "confidence": 0.95,
  "explanation": "The customer claims they never placed an order."
}
```

**Tests** (rules engine only — they do not call Claude):

```bash
pytest -v
```

## Layout

| File | Role |
|---|---|
| `disputedesk.py` | Models, action rules, Claude classifier, FastAPI `/classify` |
| `anthropic_api_call.py` | Anthropic client (reads `ANTHROPIC_API_KEY` from `.env`) |
| `test_disputedesk.py` | Pytest coverage for `recommend_action` |
| `conftest.py` | Fails fast if pytest is run on Python &lt; 3.11 |

## Version history

**v0** — Amount + delivery confirmation → a coarse label (`likely_product_not_received`, `likely_duplicate_charge`, `needs_review`).

**v0.5** — `Transaction` / `Dispute` dataclasses and `recommend_action` instead of the amount heuristic.

**v1** — Four reason codes, more synthetic cases, and pytest for contest / accept / needs_review paths. `product_unacceptable` falls through to needs_review.

**v1.1** — `Dispute.reason` is a Pydantic `Literal`, so invalid reason codes fail at construction.

**v2** — Claude classifies a customer message; Pydantic (`DraftClassification`) validates the model JSON; FastAPI + uvicorn expose `POST /classify`.
