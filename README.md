# DisputeDesk

A payment-dispute / chargeback copilot. It classifies a customer’s complaint into a Stripe/Razorpay-style reason code, retrieves evidence, and drafts a reply that cites it.

**Current milestone: v4.** Semantic RAG: embed the customer message with `all-MiniLM-L6-v2`, then nearest-neighbor search in Postgres (`pgvector`) for the closest evidence snippets. The older ID lookup (`retrieve_evidence`) is still in the file.

## What it does

Four pieces sit side by side:

1. **Recommend** (v1) — a known `Dispute` + `Transaction` → a rules-based action string.
2. **Classify** (v2) — free-text customer message → Claude → `DraftClassification` (`reason`, `confidence`, `explanation`).
3. **ID retrieve + draft** (v3.5) — customer message + transaction ID → `retrieve_evidence(id)` → Claude → `DraftResponse`.
4. **Semantic retrieve** (v4) — customer message → `retrieve_evidence_semantic(query)` → top-k evidence texts by cosine distance (`<=>`).

`retrieve_evidence` filters the in-memory `evidence_items` list by `transaction_id`.

`retrieve_evidence_semantic` encodes the query, runs `ORDER BY embedding <=> query_vector LIMIT top_k` (default `top_k=2`) against a Postgres `evidence` table, and returns the matching texts.

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

In-memory records in `disputedesk.py` (T1–T4 / D1–D4, evidence for T1–T3) still stand in for orders. Embeddings for those snippets live in Postgres once seeded (the seed `INSERT` loop is commented in `main()`).

## Setup

Python **3.11+**. The repo pins `3.11.9` in `.python-version`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install fastapi uvicorn pydantic anthropic python-dotenv
```

`pyproject.toml` already pulls in `sentence-transformers`, `psycopg2-binary`, and `pgvector`.

Create a `.env` in the project root (do not commit it):

```
ANTHROPIC_API_KEY=your_key_here
```

**Postgres + pgvector** must be running locally. The app connects as:

```
host=localhost port=5432 dbname=postgres user=postgres password=devpassword
```

`all-MiniLM-L6-v2` embeddings are 384-dimensional. The `evidence` table needs a `vector` column of that size, for example:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS evidence (
    transaction_id text,
    text text,
    embedding vector(384)
);
```

Uncomment the seed loop in `main()` once to insert the in-memory `evidence_items` rows, then comment it again so you do not duplicate them on every run.

## Run

**CLI** — current `main()` prints semantic matches for a “package never arrived” query:

```bash
python disputedesk.py
```

**API** — FastAPI app, reload on save:

```bash
uvicorn disputedesk:app --reload
```

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

**Tests** (rules engine only — they do not call Claude or Postgres):

```bash
pytest -v
```

## Layout

| File | Role |
|---|---|
| `disputedesk.py` | Models, action rules, ID retrieve, semantic retrieve (pgvector), classifier, FastAPI `/classify` |
| `anthropic_api_call.py` | Anthropic client (reads `ANTHROPIC_API_KEY` from `.env`) |
| `test_disputedesk.py` | Pytest coverage for `recommend_action` |
| `conftest.py` | Fails fast if pytest is run on Python &lt; 3.11 |

## Version history

**v0** — Amount + delivery confirmation → a coarse label (`likely_product_not_received`, `likely_duplicate_charge`, `needs_review`).

**v0.5** — `Transaction` / `Dispute` dataclasses and `recommend_action` instead of the amount heuristic.

**v1** — Four reason codes, more synthetic cases, and pytest for contest / accept / needs_review paths. `product_unacceptable` falls through to needs_review.

**v1.1** — `Dispute.reason` is a Pydantic `Literal`, so invalid reason codes fail at construction.

**v2** — Claude classifies a customer message; Pydantic (`DraftClassification`) validates the model JSON; FastAPI + uvicorn expose `POST /classify`.

**v2.5** — `DraftResponse`: classify the dispute and draft a short customer-facing reply. Markdown fences around Claude’s JSON are stripped before Pydantic validation.

**v2.7** — Simple RAG: `retrieve_evidence(transaction_id)` grabs every `EvidenceItem` for that ID (no embeddings) and injects it into the draft prompt.

**v2.8** — Semantic RAG: `SentenceTransformer("all-MiniLM-L6-v2")` + Postgres `pgvector`. `retrieve_evidence_semantic` returns the top-k nearest evidence texts by cosine distance.

**v3** — Semantic RAG + Response Drafting: `SentenceTransformer("all-MiniLM-L6-v2")` + Postgres `pgvector`. `retrieve_evidence_semantic` + `DraftResponse` classify the dispute and draft a short customer-facing reply

