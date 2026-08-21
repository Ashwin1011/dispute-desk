# DisputeDesk

A payment-dispute / chargeback copilot. A LangGraph pipeline retrieves evidence for a transaction, scores a simple fraud signal, asks Claude to classify the complaint and draft a reply that cites the evidence, a critic checks that the draft is grounded, then a human must approve before the reply is “submitted.”

**Current milestone: v3.5.** Human-in-the-loop (v3.4) plus a Postgres `audit_log` row per graph node.

## What it does (through v3.5)

`main()` invokes `app_graph` with a customer message and a transaction ID (`T1`–`T4`), then **resumes** with `Command(resume={"approved": True})` after the interrupt. Shared state is `DisputeState`:

| Field | Set by | Meaning |
|---|---|---|
| `customer_message` | input | Free-text complaint |
| `transaction_id` | input | Which order to look up (`T1` …) |
| `evidence` | `retrieve` | Top-k evidence strings from Postgres |
| `fraud_flag` | `fraud` | Amount z-score vs the fixture set |
| `draft` | `draft` (maybe rewritten by `critic`) | Reason, citations, reply, confidence |
| `critic_verdict` | `critic` | Grounded? Escalate? Notes |
| `approved` | `await_approval` | Human yes/no from `Command(resume=…)` |
| `submitted` | `submit` | Whether the reply was treated as sent |

Graph:

```
START ──► retrieve ──► draft ──► critic ──► await_approval ──► submit ──► END
       └► fraud    ──►
```

`retrieve` and `fraud` run from `START` in parallel; both must finish before `draft`. After the critic, the graph **pauses** until a human resumes it.

### Retrieve (`retrieve_node` / v3.2)

`retrieve_evidence_semantic(query, transaction_id, top_k=2)`:

1. Embeds the customer message with `SentenceTransformer("all-MiniLM-L6-v2")` (384-d).
2. Queries Postgres `evidence` with pgvector cosine distance (`<=>`), **filtered to that `transaction_id`**.
3. Returns the closest `top_k` texts.

Earlier (v2.7) this was in-memory “grab every `EvidenceItem` for this ID” with no embeddings. That helper is commented out.

### Fraud (`fraud_node` / v3.3)

`check_fraud_signals(transaction_id)` looks up the fixture `Transaction` and computes a z-score of its `amount` against all fixture amounts (`statistics.mean` / `pstdev`). If `|z| > 1.5`, `is_anomaly` is true and the note says the amount is flagged for review. Four rows (`T1` ₹2499, `T2` ₹1200, `T3` ₹50, `T4` ₹800) — T1 is the outlier.

Output is `FraudFlag(z_score, is_anomaly, note)`.

### Draft (`draft_node` / v3–v3.2)

`draft_from_evidence` sends the message plus retrieved bullets to Claude (`claude-sonnet-4-6`). The model must return JSON only:

```json
{
  "reason": "unrecognized | product_not_received | duplicate | product_unacceptable",
  "evidence_cited": ["..."],
  "draft_text": "...",
  "confidence": 0.0
}
```

Markdown fences around the JSON are stripped (`parse_model_json`). Only Anthropic `TextBlock`s are read (`message_text`). Pydantic (`DraftResponse`) validates the object.

### Critic (`critic_node` / v3.3)

`check_grounding` requires every `evidence_cited` string to appear in the retrieved evidence (normalized: trim, drop trailing `.`, lowercase, substring either way).

- If not grounded: call `draft_from_evidence` **once** and re-check.
- Escalate (`escalate_for_review`) if the fraud flag is an anomaly **or** the draft is still ungrounded after retry.

`CriticVerdict`: `grounded`, `escalate_for_review`, `notes`.

### Human approval (`await_approval_node` / v3.4)

The graph is compiled with `MemorySaver` and a `thread_id` (`dispute-T1-demo` in `main()`). `await_approval_node` calls LangGraph `interrupt({…})` with the draft text, critic notes, and `escalate_for_review`. The first `invoke` returns in a paused state (`PAUSED` in `main()`).

Resume with:

```python
app_graph.invoke(Command(resume={"approved": True}), config=config)
```

`approved` is stored on state. Without a checkpointer, interrupt/resume cannot continue the same run.

### Submit (`submit_node` / v3.4)

If `approved` is true, prints `[SUBMITTED] response for {transaction_id}` and sets `submitted=True`. Otherwise prints `[REJECTED]` and sets `submitted=False`. There is no real PSP/network submit yet — this is the stand-in for “send the reply.”

### Audit log (`log_decision` / v3.5)

Every node writes a JSON decision blob to Postgres `audit_log` (`thread_id`, `transaction_id`, `node_name`, `decision`): retrieve (evidence texts), fraud (z-score), draft (reason + reply), critic (verdict), await_approval (approved), submit (submitted). Use the same `thread_id` as the LangGraph checkpointer (`dispute-T1-demo` in `main()`).

Example table:

```sql
CREATE TABLE IF NOT EXISTS audit_log (
    id serial PRIMARY KEY,
    thread_id text,
    transaction_id text,
    node_name text,
    decision jsonb,
    created_at timestamptz DEFAULT now()
);
```

### Reason codes

| Reason | Meaning |
|---|---|
| `unrecognized` | Customer does not recognize the charge |
| `product_not_received` | Paid, but nothing arrived |
| `duplicate` | Charged more than once |
| `product_unacceptable` | Received, but wrong / damaged / not as described |

### Still in the file, not on the live path

v1 `recommend_action` (contest / accept / needs_review from delivery flags) and v2 `classify_dispute` + FastAPI `POST /classify` are commented out. Pytest still targets `recommend_action` and will fail until that function is restored or the tests are updated.

Fixture transactions T1–T4 remain in memory. Evidence embeddings live in Postgres after a one-time seed (the `INSERT` loop in `main()` is commented).

## Python environment (`.venv`)

This machine has several Pythons: pyenv’s global **2.7**, a system **python3**, and this project’s **`.venv`**. Packages like `langgraph` live only in `.venv`. Bare `python disputedesk.py` uses pyenv, not the venv, which is why you see `ModuleNotFoundError`.

A **virtual environment** is an isolated Python for this repo — its own interpreter and installed packages, so this project cannot clash with other projects (or with Python 2.7). `.venv` is that folder; it is gitignored.

Use one of these, every time:

```bash
source .venv/bin/activate    # prompt shows (.venv)
python disputedesk.py
```

```bash
./run                       # always calls .venv/bin/python
```

```bash
.venv/bin/python disputedesk.py
```

`deactivate` leaves the venv. Cursor is set to use `.venv/bin/python` as the interpreter; pick that if the editor asks.

Optional: make even bare `python` in this folder point at the venv (pyenv):

```bash
ln -sfn "$(pwd)/.venv" ~/.pyenv/versions/dispute-desk
pyenv local dispute-desk
```

## Setup

Python **3.11+**. One-time:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`pyproject.toml` is the source of truth (`langgraph`, `anthropic`, `sentence-transformers`, `psycopg2-binary`, `pgvector`, FastAPI/uvicorn, pytest). Never `pip install` into the global 2.7 `python`.

Create a `.env` in the project root (do not commit it):

```
ANTHROPIC_API_KEY=your_key_here
```

**Postgres + pgvector** must be running locally. The app connects as:

```
host=localhost port=5432 dbname=postgres user=postgres password=devpassword
```

`all-MiniLM-L6-v2` embeddings are 384-dimensional:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS evidence (
    transaction_id text,
    text text,
    embedding vector(384)
);
```

Uncomment the seed loop in `main()` once to insert evidence rows, then comment it again so you do not duplicate them.

## Run

**CLI** — `main()` runs retrieve ∥ fraud → draft → critic, **pauses** for approval, then resumes with `approved: True` and submit:

```bash
./run
# or
source .venv/bin/activate && python disputedesk.py
```

You should see `PAUSED —` then `RESUMED —` (and `[SUBMITTED]` if approval was true).

**API** — `POST /classify` is commented out. To serve it again, uncomment the FastAPI route and:

```bash
uvicorn disputedesk:app --reload
```

**Tests** (v1 rules engine; currently import `recommend_action`, which is commented):

```bash
pytest -v
```

## Layout

| File | Role |
|---|---|
| `disputedesk.py` | Models, pgvector retrieve, fraud z-score, Claude draft, critic, HITL interrupt/submit, LangGraph |
| `anthropic_api_call.py` | Anthropic client (`ANTHROPIC_API_KEY` from `.env`) |
| `run` | Wrapper that always uses `.venv/bin/python` |
| `test_disputedesk.py` | Pytest for `recommend_action` (v1) |
| `conftest.py` | Fails fast if pytest is run on Python &lt; 3.11 |

## Version history

**v0** — Amount + delivery confirmation → a coarse label (`likely_product_not_received`, `likely_duplicate_charge`, `needs_review`).

**v0.5** — `Transaction` / `Dispute` dataclasses and `recommend_action` instead of the amount heuristic.

**v1** — Four reason codes, synthetic cases, pytest for contest / accept / needs_review. `product_unacceptable` falls through to needs_review.

**v1.1** — `Dispute.reason` is a Pydantic `Literal`, so invalid reason codes fail at construction.

**v2** — Claude classifies a customer message; Pydantic (`DraftClassification`) validates the JSON; FastAPI + uvicorn expose `POST /classify`.

**v2.5** — `DraftResponse`: classify and draft a short customer-facing reply. Markdown fences stripped before Pydantic validation.

**v2.7** — Simple RAG: `retrieve_evidence(transaction_id)` grabs every in-memory `EvidenceItem` for that ID (no embeddings) and injects it into the draft prompt.

**v2.8 / v2.9** — Semantic RAG: MiniLM embeddings in Postgres (`pgvector`). `retrieve_evidence_semantic` returns top-k nearest texts; draft uses those hits (`v2.9`).

**v3** — Semantic retrieve + `DraftResponse` wired together as the main path (v3 complete).

**v3.1** — LangGraph: `DisputeState` and a compiled graph with a single `draft` node.

**v3.2** — `retrieve_node` split from `draft_node`. Retrieve is semantic search **scoped to `transaction_id`**. Graph: `START → retrieve → draft → END`.

**v3.3** — `fraud_node` (amount z-score, `|z| > 1.5` → anomaly) in parallel with retrieve; both feed `draft`. `critic_node` checks citations against retrieved evidence, retries the draft once if ungrounded, and sets `escalate_for_review` when there is a fraud anomaly or the draft still is not grounded.

**v3.4** — Human-in-the-loop. After critic: `await_approval` (`interrupt` with draft + critic notes), `MemorySaver` checkpointer + `thread_id`, resume via `Command(resume={"approved": True|False})`, then `submit_node` marks the reply submitted or rejected. Demo in `main()` pauses, then auto-resumes with approval.

**v3.5** — Durable audit trail. `log_decision` inserts one `audit_log` row per node (retrieve, fraud, draft, critic, await_approval, submit) keyed by `thread_id` + `transaction_id`.
