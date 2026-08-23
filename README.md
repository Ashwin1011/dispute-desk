# DisputeDesk

A payment-dispute / chargeback copilot. A LangGraph pipeline retrieves evidence for a transaction, scores a simple fraud signal, asks Claude to classify the complaint and draft a reply that cites the evidence, and a critic checks that the draft is grounded. A human is asked to approve **only when** the critic escalates; otherwise submit is automatic. The same graph is exposed as an MCP server so Cursor (or any MCP client) can run and approve disputes as tools.

Evidence, fraud scoring, and MCP calls are **scoped to a tenant**. A tenant cannot retrieve another tenant's evidence or score fraud on another tenant's transactions.

**Current milestone:** the full `retrieve ∥ fraud → draft → critic → conditional HITL → submit → audit-log` pipeline is built, checkpointed, and shipped as an MCP server — verified live from Cursor, including a real human-rejection path. Retrieval and fraud checking are tenant-scoped, and retrieval is now **hybrid** (Postgres full-text keyword search fused with pgvector semantic search via Reciprocal Rank Fusion) rather than semantic-only — all three retrieval methods, plus fraud checking, are covered by a 19/19-passing automated cross-tenant test suite. See *Known limitations* and *What's next* below before treating any part of this as finished.

> **Note on version numbers below:** the `v0`–`v5.x` tags in this file are this project's own chronological build order, not the phase numbers from the project's planning docs. There, `v4` names the whole multi-agent + MCP milestone, and `v5` names a golden-dataset eval + CI-gated regression + formal red-team suite that **has not been started**. What this file calls `v5` (tenant isolation) is really closing a permission-boundary gap the original plan scoped under `v3`. Flagging this so the two don't read as contradictory.

## What it does (through v5)

`main()` invokes `app_graph` with a customer message, a tenant, and a transaction. If the critic escalates, the run **pauses** and `main()` resumes with `Command(resume={"approved": True})`. If it does not escalate, the first `invoke` finishes through `submit` (auto-approved) and there is nothing to resume. Shared state is `DisputeState`:

| Field              | Set by                                | Meaning                               |
| ------------------ | ------------------------------------- | ------------------------------------- |
| `customer_message` | input                                 | Free-text complaint                   |
| `tenant_id`        | input                                 | Which merchant owns the order         |
| `transaction_id`   | input                                 | Which order to look up (`T1` …)       |
| `evidence`         | `retrieve`                            | Top-k evidence strings from Postgres  |
| `fraud_flag`       | `fraud`                               | Amount z-score vs that tenant's txs   |
| `draft`            | `draft` (maybe rewritten by `critic`) | Reason, citations, reply, confidence  |
| `critic_verdict`   | `critic`                              | Grounded? Escalate? Notes             |
| `approved`         | `await_approval`                      | Human yes/no from `Command(resume=…)` |
| `submitted`        | `submit`                              | Whether the reply was treated as sent |

Graph:

```
START ──► retrieve ──► draft ──► critic ──┬── (escalate_for_review) ──► await_approval ──► submit ──► END
       └► fraud    ──►                   └── (else) ─────────────────► submit ──────────► END
```

`retrieve` and `fraud` run from `START` in parallel; both must finish before `draft`. After the critic, `route_after_critic` sends the run to `await_approval` only when `escalate_for_review` is true; otherwise it skips the human and goes to `submit`.

### Retrieve (`retrieve_node` / v3.2, tenant-scoped in v5, hybrid in v5.1)

`retrieve_node` calls `retrieve_evidence_hybrid`, which combines two independently tenant-scoped retrieval methods rather than relying on either alone:

**Semantic** — `retrieve_evidence_semantic(query, tenant_id, transaction_id, top_k)`. Embeds the query with `SentenceTransformer("all-MiniLM-L6-v2")` (384-d), ranks by pgvector cosine distance (`<=>`), filtered to `tenant_id` + `transaction_id`. Always returns *something* ranked "closest," even when nothing is actually that close — no requirement that any literal words overlap.

**Keyword** — `retrieve_evidence_keyword(query, tenant_id, transaction_id, top_k)`. `evidence.text_search` is a generated `tsvector` column (`GENERATED ALWAYS AS (to_tsvector('english', text)) STORED`, GIN-indexed), auto-derived from `text` on every insert. A query is normalized the same way via `plainto_tsquery`, matched with the `@@` operator, and ranked by `ts_rank`. Unlike semantic search, this is a **hard filter first** — a row only comes back at all if it shares an actual word (lexeme) with the query; there's no partial credit for "topically related."

**Fusion** — `retrieve_evidence_hybrid(query, tenant_id, transaction_id, top_k, rrf_k=60)`. Pulls `top_k=10` from each method independently, then combines via Reciprocal Rank Fusion: each result's score is `1/(rrf_k + rank)` in whichever list(s) it appears, summed across both lists, then re-sorted and truncated to the real `top_k`. RRF fuses by *rank position*, not raw score — cosine distance and `ts_rank` live on incomparable scales, so averaging them directly wouldn't mean anything.

**Verified, not assumed to help:** ran a query about an exact identifier ("what about tracking TRK556") against real fixture data with three T1 evidence rows. Keyword search returned only the row containing the literal string `TRK556` — a second, genuinely delivery-related row ("order confirmation email...") shares zero words with the query, so it never passed the `@@` filter at all, not just ranked lower. Semantic search returned both, since it has no word-overlap requirement. Hybrid correctly surfaced both, tracking-number evidence first. A specific, reproducible divergence — not a description of what hybrid search is supposed to do in theory.

**Deliberately not BM25.** Postgres's built-in `ts_rank` predates BM25 and lacks its two key improvements — term-frequency saturation and document-length normalization — both of which matter most for large, variable-length document collections. DisputeDesk's evidence is uniformly short, single-sentence snippets, where those advantages are largely theoretical rather than practical. Documented as a reasoned trade-off, not an oversight: if evidence documents ever grow longer and more variable (full email threads, multi-paragraph tickets), the concrete next move is the ParadeDB `pg_search` Postgres extension (real BM25 via Tantivy, no new infrastructure to run) or a dedicated search engine — not needed for the data this project actually has today.

Earlier (v2.7) retrieval was in-memory "grab every `EvidenceItem` for this ID" with no embeddings. That helper is commented out.

### Fraud (`fraud_node` / v3.3, tenant-scoped in v5)

`check_fraud_signals(tenant_id, transaction_id)` looks up the fixture `Transaction` **among that tenant's rows only** and computes a z-score of its `amount` against that tenant's amounts (`statistics.mean` / `pstdev`). If the id is missing for that tenant, it returns `note="transaction not found for this tenant"` and does not score another merchant's order.

If `|z| > 1.5`, `is_anomaly` is true and the note says the amount is flagged for review.

Fixtures:

| ID | Tenant        | Amount |
| -- | ------------- | ------ |
| T1 | electromart   | ₹2499  |
| T2 | electromart   | ₹1200  |
| T3 | subscribebox  | ₹50    |
| T4 | subscribebox  | ₹800   |

Output is `FraudFlag(z_score, is_anomaly, note)`.

**Worth knowing:** before tenant scoping, T1's amount looked anomalous because it was being compared against *every* tenant's transactions combined, including subscribebox's much smaller ones. Scoping the z-score to electromart alone was the correct fix — but with exactly two transactions per tenant, it also means `|z|` is now mathematically always exactly `1.0` for either transaction in a tenant (the z-score of either point in a 2-point population is always `±1`, regardless of the actual amounts), so `is_anomaly` currently can never fire from amount alone. See *Known limitations*.

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

Given the fraud-anomaly path currently can't fire (see above), the grounding-failure path is, for now, the only way `escalate_for_review` actually triggers in this fixture set — confirmed by requesting a cross-tenant transaction: `evidence` comes back empty, the draft cites nothing real, `grounded=False`, and the critic escalates on its own, independent of the fraud flag. Two separate safety nets, both real.

### Human approval (`await_approval_node` / v3.4, routed in v3.6)

Reached **only if** the critic set `escalate_for_review` (fraud anomaly or still-ungrounded draft). The graph is compiled with `MemorySaver` and a `thread_id` (`dispute-T1-demo` in `main()`; MCP uses `dispute-{tenant_id}-{transaction_id}`). `await_approval_node` calls LangGraph `interrupt({…})` with the draft text, critic notes, and `escalate_for_review`. That `invoke` returns paused (`PAUSED` in `main()`).

Resume with:

```python
app_graph.invoke(Command(resume={"approved": True}), config=config)
```

`approved` is stored on state. Without a checkpointer, interrupt/resume cannot continue the same run. If the critic did not escalate, this node never runs.

### Submit (`submit_node` / v3.4, auto vs human in v3.6)

- Critic escalated: `submitted` follows the human `approved` flag. Audit `approval_source` is `human` or `human_rejected`.
- Critic did not escalate: `submitted=True` with `approval_source=auto` (no interrupt).

Prints `[SUBMITTED] … (auto|human)` or `[REJECTED]`. There is no real PSP/network submit yet — this is the stand-in for "send the reply."

### Audit log (`log_decision` / v3.5)

Every node writes a JSON decision blob to Postgres `audit_log` (`thread_id`, `transaction_id`, `node_name`, `decision`): retrieve (tenant + evidence texts), fraud (tenant + z-score), draft (reason + reply), critic (verdict), await_approval (approved, if HITL ran), submit (`submitted` + `approval_source`). Use the same `thread_id` as the LangGraph checkpointer.

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

### MCP (`mcp_server.py` / v4, `tenant_id` in v5)

`MCPServer("DisputeDesk")` from `mcp[cli]>=2.0` exposes the compiled graph as two stdio tools. The graph itself is unchanged; this is an interface in front of `app_graph.invoke`.

| Tool              | Arguments                                      | What it does                                                                                                                                                                                              |
| ----------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `resolve_dispute` | `tenant_id`, `transaction_id`, `customer_message` | `invoke` with `thread_id=dispute-{tenant_id}-{transaction_id}`. If the graph interrupts, returns `status=awaiting_approval` plus `draft_text` / `critic_notes`. Otherwise `submitted` or `rejected`.   |
| `approve_dispute` | `thread_id`, `approved`                        | `Command(resume={"approved": …})` on that thread. Returns `submitted` or `rejected` and the draft text.                                                                                                   |

Example: resolve T1 for tenant `electromart` with `"I never received this order."` Clean cases finish in one `resolve_dispute` call; escalate cases return `awaiting_approval` and need `approve_dispute` with the returned `thread_id`. A request naming a tenant that doesn't own the transaction (e.g. tenant `subscribebox` with `transaction_id=T1`) correctly returns no evidence and escalates via the grounding check — verified live, not just in tests.

Cursor is configured in `.cursor/mcp.json`. The command is this project's **`.venv/bin/python`** with an absolute path to `mcp_server.py` — not bare `uv` / `python`. Cursor's GUI process does not inherit the shell PATH, so `spawn uv ENOENT` is what you get if the command is just `uv`, and `ECONNREFUSED ::1:8000` is what you get if the config accidentally points at an SSE/URL transport instead of a local stdio command.

### Reason codes

| Reason                 | Meaning                                          |
| ---------------------- | ------------------------------------------------ |
| `unrecognized`         | Customer does not recognize the charge           |
| `product_not_received` | Paid, but nothing arrived                        |
| `duplicate`            | Charged more than once                           |
| `product_unacceptable` | Received, but wrong / damaged / not as described |

### Still in the file, not on the live path

v1 `recommend_action` (contest / accept / needs_review from delivery flags) and v2 `classify_dispute` + FastAPI `POST /classify` are commented out. `tests/test_disputedesk.py` still targets `recommend_action` and will fail until that function is restored or those tests are updated.

Fixture transactions T1–T4 remain in memory (each with a `tenant_id`). Evidence embeddings live in Postgres after a one-time seed (`uv run python seed_evidence.py`).

## Known limitations

Named here on purpose rather than left implicit — these are the honest edges of what's built so far, not things papered over:

- **The fraud-anomaly escalation path can't currently fire.** With exactly two transactions per tenant, `check_fraud_signals`'s z-score is mathematically always `±1.0` (true for any 2-point population regardless of the actual amounts), which never crosses the `1.5` threshold. `is_anomaly` is real code, correctly scoped, but currently unreachable by the fixture data. A third transaction per tenant with a genuine outlier amount would restore a meaningful demo of that path — not done yet.
- **`MemorySaver` is dev-only.** It doesn't survive a process restart, so an interrupted dispute can't currently be approved in a separate run from the one that started it — only within one live Python process. A persistent checkpointer (SQLite- or Postgres-backed) is a small, well-understood swap for later, not built yet.
- **LangGraph logs a deserialization warning** for the custom Pydantic types (`FraudFlag`, `DraftResponse`, `CriticVerdict`) stored in checkpoints — a real, forward-looking security signal (LangGraph is moving toward blocking arbitrary-class deserialization from checkpoints by default). Deferred; the clean fix is storing `.model_dump()` dicts in state instead of raw model instances, not whitelisting class names.
- **`audit_log` has a narrow theoretical race.** `retrieve` and `fraud` run in parallel and both write to `audit_log` through the same shared `conn` (each opens its own cursor, which mitigates but doesn't fully eliminate risk under a truly concurrent executor). Worth a connection pool before this is anything more than a demo.
- **No customer-level fraud memory.** `FraudAnalyst` only looks at one transaction's amount, tenant-scoped — it has no notion of "this customer has filed three disputes this month," which is itself a real fraud signal. Not built.
- **Data is entirely synthetic**, by design — four fixture transactions, three evidence rows, two tenants. Good enough to prove the architecture; not a claim about handling real volume or variety.

## What's next

Per the original build plan: hybrid search (Postgres full-text + pgvector) is **done** — `retrieve_node` now calls `retrieve_evidence_hybrid`, verified against a real keyword-vs-semantic divergence rather than just described. Still pending from that same phase: a documented **Weaviate comparison experiment** (same evidence, measured retrieval quality/latency against the current pgvector + Postgres-full-text setup — a genuine "did switching vector stores help, and by how much" writeup, not a replacement). After that: Tavily for looking up current dispute policy externally, then the project's actual `v5` — a golden-dataset eval, CI-gated regression, and a formal red-team suite (the permission-boundary tests above are a first real piece of that, worth folding in rather than rebuilding) — and `v6`, production deployment with monitoring and a maintenance runbook.

## Python environment (`.venv`)

This machine has several Pythons: pyenv's global **2.7**, a system **python3**, and this project's **`.venv`**. Packages like `langgraph` live only in `.venv`. Bare `python disputedesk.py` uses pyenv, not the venv, which is why you see `ModuleNotFoundError`.

The same trap hits tests: bare `pytest` (or `uv run pytest` **before** pytest is in the project env) can pick up `/Library/Frameworks/Python.framework/.../python3`, which does not have `langgraph`. Always run tests with `uv run pytest` after `uv sync`, so pytest is `.venv/bin/pytest`.

A **virtual environment** is an isolated Python for this repo — its own interpreter and installed packages, so this project cannot clash with other projects (or with Python 2.7). `.venv` is that folder; it is gitignored.

Use one of these, every time:

```bash
uv run python disputedesk.py
uv run pytest tests/test_permission_boundary.py -v
```

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

Python **3.11+**. One-time (preferred):

```bash
uv sync
```

That installs runtime deps plus the default `dev` group (`pytest`) into `.venv`. `pyproject.toml` is the source of truth (`langgraph`, `anthropic`, `sentence-transformers`, `psycopg2-binary`, `pgvector`, FastAPI/uvicorn, `mcp[cli]`, pytest). Never `pip install` into the global 2.7 `python`.

Pip alternative:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

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
    tenant_id text,
    transaction_id text,
    text text,
    embedding vector(384)
);
```

Seed evidence once (do not re-run without clearing rows, or you will duplicate them):

```bash
uv run python seed_evidence.py
```

## Run

**CLI** — `main()` runs electromart T1: retrieve ∥ fraud → draft → critic, then either auto-submit or pause + resume:

```bash
uv run python disputedesk.py
# or
./run
# or
source .venv/bin/activate && python disputedesk.py
```

With the current fixture data, T1 and T2 (electromart) and T3 and T4 (subscribebox) all draft and auto-submit without pausing — the fraud-anomaly path can't fire yet (see *Known limitations*). To actually see a pause + `Command(resume=…)` cycle right now, request a transaction under the *wrong* tenant (e.g. tenant `subscribebox`, transaction `T1`): retrieval correctly comes back empty, the draft can't ground itself, and the critic escalates. The demo still calls `Command(resume=…)` after the first invoke; that only actually does something if the graph paused — resuming an already-completed thread is a safe no-op.

**MCP** — Cursor loads `.cursor/mcp.json` and spawns `mcp_server.py` over stdio. Ask the agent to resolve a dispute for a tenant (e.g. electromart T1 with `"I never received this order."`). Clean cases finish in one `resolve_dispute` call; escalate cases return `awaiting_approval` and need `approve_dispute` with the returned `thread_id`.

To run the server by hand (same interpreter the MCP config uses):

```bash
.venv/bin/python mcp_server.py
```

**API** — `POST /classify` is commented out. To serve it again, uncomment the FastAPI route and:

```bash
uv run uvicorn disputedesk:app --reload
```

**Tests** — tenant isolation across every retrieval method (live path). Needs Postgres with seeded evidence:

```bash
uv run pytest tests/test_permission_boundary.py -v
```

19 cases: every transaction tried against every tenant that doesn't own it, across all three retrieval methods (semantic, keyword, hybrid) plus the fraud check, plus five positive sanity checks proving the boundary blocks wrong tenants without blocking everyone. The keyword/hybrid cross-tenant cases deliberately build their attack query from the target transaction's own evidence text — a generic query would return empty regardless of whether the tenant boundary works at all (keyword search's hard match gate means "no shared words" and "correctly blocked" look identical), so the test needs a query that *would* match if the boundary were actually broken. Extending past semantic-only search this way caught a real, if minor, gap: T4 had a `Transaction` fixture but no matching `EvidenceItem` at all, which crashed the test before it even reached the boundary logic — closed by adding T4's missing evidence rather than working around the test.

`tests/test_disputedesk.py` still imports v1 `recommend_action` (commented out) and is not part of the live path.

## Layout

| File                                 | Role                                                                                             |
| ------------------------------------ | ------------------------------------------------------------------------------------------------ |
| `disputedesk.py`                     | Models, tenant-scoped pgvector retrieve, fraud z-score, Claude draft, critic, HITL, LangGraph    |
| `mcp_server.py`                      | MCP stdio server: `resolve_dispute` / `approve_dispute` wrap `app_graph`                         |
| `seed_evidence.py`                   | One-time insert of fixture evidence rows (tenant_id + embedding) into Postgres                   |
| `.cursor/mcp.json`                   | Cursor MCP config (absolute `.venv/bin/python` + `mcp_server.py`)                                |
| `anthropic_api_call.py`              | Anthropic client (`ANTHROPIC_API_KEY` from `.env`)                                               |
| `run`                                | Wrapper that always uses `.venv/bin/python`                                                      |
| `tests/test_permission_boundary.py`  | Pytest: wrong tenant gets no evidence / no fraud signal; correct tenant still retrieves          |
| `tests/test_disputedesk.py`          | Pytest for `recommend_action` (v1, not on the live path)                                         |
| `conftest.py`                        | Fails fast if pytest is run on Python &lt; 3.11                                                  |

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

**v3.6** — Conditional HITL. `route_after_critic` sends escalate cases to `await_approval` and clean cases straight to `submit`. `submit_node` records `approval_source` as `auto`, `human`, or `human_rejected`.

**v4** — MCP server. `mcp_server.py` (`MCPServer`, stdio) exposes `resolve_dispute` and `approve_dispute` over the same `app_graph`. Cursor config is `.cursor/mcp.json` using `.venv/bin/python` so the GUI spawn does not depend on `uv` being on PATH. Verified live from Cursor's Agent chat, including a real human-rejection path. (v4 complete.)

**v5** — Multi-tenant isolation, closing a permission-boundary gap from the original `v3` spec rather than the roadmap's actual `v5` (see the note at the top of this file). `tenant_id` on `Transaction`, `EvidenceItem`, `DisputeState`, Postgres `evidence`, retrieve (`WHERE tenant_id AND transaction_id`), and fraud (z-score against that tenant's amounts only; unknown pair → `"transaction not found for this tenant"`). MCP `resolve_dispute` takes `tenant_id`; thread ids are `dispute-{tenant_id}-{transaction_id}`. Seed via `seed_evidence.py`. Found and fixed a real bug this exposed: pre-fix, the fraud z-score was computed across all tenants combined, which was both a soft data leak and statistically wrong — fixing it changed T1's fraud verdict from anomalous to normal. Permission-boundary tests in `tests/test_permission_boundary.py` (9/9 passing: every transaction against every tenant that doesn't own it, plus a positive sanity check). Run tests with `uv run pytest` so pytest uses `.venv`, not system Python.

**v5.1** — Hybrid retrieval. Added Postgres full-text search (`text_search` generated `tsvector` column + GIN index) as `retrieve_evidence_keyword`, alongside the existing `retrieve_evidence_semantic`. Combined via `retrieve_evidence_hybrid` using Reciprocal Rank Fusion (`1/(60+rank)` per method, summed across both lists) rather than raw-score averaging, since cosine distance and `ts_rank` aren't on comparable scales. `retrieve_node` now calls the hybrid function, not semantic alone. Deliberately did not implement BM25 — Postgres's `ts_rank` predates it and lacks BM25's term-frequency saturation and document-length normalization, but those matter most for long, variable-length documents, and this project's evidence is short, uniform-length snippets; a real BM25 extension (ParadeDB `pg_search`) is the concrete answer if that stops being true. Verified with a real comparison, not an assumed one: a query about an exact identifier (`TRK556`) diverged meaningfully between keyword-only (hard filter, missed a genuinely related but differently-worded row entirely) and semantic-only (no word-overlap requirement, caught it) — hybrid correctly surfaced both. Extended `tests/test_permission_boundary.py` to cover the two new retrieval methods (9 → 19 cases), using attack queries built from the target transaction's own evidence text so a passing test actually proves the boundary works rather than coincidentally returning nothing. That extension caught a real fixture gap: T4 had a `Transaction` but no matching `EvidenceItem`, crashing the test before it reached any tenant-boundary logic — fixed by adding T4's evidence rather than working around the test. (v5.1 complete — see *Known limitations* and *What's next* for what the roadmap's real v5/v6 still require.)
