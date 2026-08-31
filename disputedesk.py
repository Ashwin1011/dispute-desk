#!/usr/bin/env python
from dataclasses import dataclass
from datetime import date
from typing import Literal, TypedDict, Optional
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt, Command
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
from anthropic_api_call import client
from anthropic.types import Message, TextBlock
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import statistics
import json
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter
from tavily import TavilyClient
import os
from dotenv import load_dotenv
import atexit


load_dotenv()

from langfuse.langchain import CallbackHandler
langfuse_handler = CallbackHandler()

MODEL_STRONG = "claude-sonnet-4-6"
MODEL_CHEAP = "claude-haiku-4-5-20251001"

tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
import time

weaviate_client = weaviate.connect_to_local(host=os.environ.get("WEAVIATE_HOST", "localhost"))
atexit.register(weaviate_client.close)
app = FastAPI()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_db_uri() -> str:
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    return (
        f"postgresql://{os.environ.get('DB_USER', 'postgres')}:"
        f"{os.environ.get('DB_PASSWORD', 'devpassword')}@"
        f"{os.environ.get('DB_HOST', 'localhost')}:"
        f"{os.environ.get('DB_PORT', '5432')}/"
        f"{os.environ.get('DB_NAME', 'postgres')}"
    )

DB_URI = build_db_uri()
conn = psycopg2.connect(DB_URI)

register_vector(conn)  # teaches psycopg2 how to send Python vectors to Postgres's vector column

DB_URI = (
    f"postgresql://{os.environ.get('DB_USER', 'postgres')}:"
    f"{os.environ.get('DB_PASSWORD', 'devpassword')}@"
    f"{os.environ.get('DB_HOST', 'localhost')}:"
    f"{os.environ.get('DB_PORT', '5432')}/"
    f"{os.environ.get('DB_NAME', 'postgres')}"
)

@dataclass
class Transaction:
    id: str
    tenant_id: str
    amount: float
    order_date: date
    delivered: bool
    delivery_address_matches_billing: bool

def log_decision(thread_id: str, transaction_id: str, node_name: str, decision: dict) -> None:
    with conn.cursor() as log_cur:
        log_cur.execute(
            """
            INSERT INTO audit_log (thread_id, transaction_id, node_name, decision)
            VALUES (%s, %s, %s, %s)
            """,
            (thread_id, transaction_id, node_name, json.dumps(decision)),
        )
    conn.commit()

def make_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}, "callbacks": [langfuse_handler]}

def get_thread_id(config: RunnableConfig) -> str:
    configurable = config.get("configurable")
    assert configurable is not None, "graph must be invoked with a configurable thread_id"
    thread_id = configurable.get("thread_id")
    assert thread_id is not None, "graph must be invoked with a configurable thread_id"
    return thread_id

class Dispute(BaseModel):
    id: str
    transaction_id: str
    reason: Literal["unrecognized", "product_not_received", "duplicate", "product_unacceptable"]


class DraftClassification(BaseModel):
    reason: Literal["unrecognized", "product_not_received", "duplicate", "product_unacceptable"]
    confidence: float
    explanation: str

class ClassifyRequest(BaseModel):
    customer_message: str

@dataclass
class EvidenceItem:
    tenant_id: str
    transaction_id: str
    text: str

evidence_items = [
    EvidenceItem("electromart", "T1", "Courier tracking TRK556 shows delivered Aug 9, signed at the billing address."),
    EvidenceItem("electromart", "T1", "Order confirmation email was sent to the customer on Aug 3 with estimated delivery Aug 9-11."),
    EvidenceItem("electromart", "T1", "Customer's account has two prior disputes marked as customer error, both resolved without refund."),
    EvidenceItem("electromart", "T2", "No delivery confirmation on file for this order."),
    EvidenceItem("subscribebox", "T3", "Order history shows only one charge of ₹50 on this date — no duplicate found."),
    EvidenceItem("subscribebox", "T4", "Delivery confirmed via courier tracking SBX882, signed for at the customer's registered address on Aug 15."),
    EvidenceItem("gizmohub", "T5", "Screen protector shipped via CourierX, delivered Aug 6, confirmed by tracking GZ1001."),
    EvidenceItem("gizmohub", "T6", "USB-C cable order confirmed delivered Aug 7 per tracking GZ1002."),
    EvidenceItem("gizmohub", "T7", "Phone case delivery confirmed Aug 8, signed for at billing address, tracking GZ1003."),
    EvidenceItem("gizmohub", "T8", "Premium wireless earbuds order placed Aug 10 — no delivery confirmation on file yet, shipment still in transit per carrier tracking GZ1004."),
]

# a tiny pretend database, standing in for real records
transactions = [
    Transaction("T1","electromart", 2499, date(2026, 8, 5), delivered=True, delivery_address_matches_billing=True),
    Transaction("T2","electromart", 1200, date(2026, 8, 10), delivered=False, delivery_address_matches_billing=False),
    Transaction("T3","subscribebox", 50, date(2026, 8, 12), delivered=True, delivery_address_matches_billing=True),
    Transaction("T4","subscribebox", 800, date(2026, 8, 15), delivered=True, delivery_address_matches_billing=True),
    Transaction("T5", "gizmohub", 300, date(2026, 8, 6), delivered=True, delivery_address_matches_billing=True),
    Transaction("T6", "gizmohub", 280, date(2026, 8, 7), delivered=True, delivery_address_matches_billing=True),
    Transaction("T7", "gizmohub", 320, date(2026, 8, 8), delivered=True, delivery_address_matches_billing=True),
    Transaction("T8", "gizmohub", 8500, date(2026, 8, 10), delivered=False, delivery_address_matches_billing=False),
    ]

def get_transaction(tenant_id: str, transaction_id: str) -> Transaction | None:
    tenant_txs = [t for t in transactions if t.tenant_id == tenant_id]
    return next((t for t in tenant_txs if t.id == transaction_id), None)

class FraudFlag(BaseModel):
    z_score: float
    is_anomaly: bool
    note: str

def check_fraud_signals(tenant_id: str, transaction_id: str) -> FraudFlag:
    tx = get_transaction(tenant_id, transaction_id)
    if tx is None:
        return FraudFlag(z_score=0.0, is_anomaly=False, note="transaction not found for this tenant")

    tenant_txs = [t for t in transactions if t.tenant_id == tenant_id]
    amounts = [t.amount for t in tenant_txs]
    mean = statistics.mean(amounts)
    stdev = statistics.pstdev(amounts) or 1.0  # guard divide-by-zero if all amounts match

    z = (tx.amount - mean) / stdev
    is_anomaly = abs(z) > 1.5  # arbitrary threshold — tune once you have more than 4 fixture rows

    note = (
        f"amount ₹{tx.amount} is {abs(z):.2f} std devs from the mean (₹{mean:.2f}) — flagged for review"
        if is_anomaly
        else f"amount ₹{tx.amount} is within normal range (z={z:.2f})"
    )
    return FraudFlag(z_score=z, is_anomaly=is_anomaly, note=note)


class CriticVerdict(BaseModel):
    grounded: bool
    escalate_for_review: bool
    notes: str

class DraftResponse(BaseModel):
    reason: Literal["unrecognized", "product_not_received", "duplicate", "product_unacceptable"]
    evidence_cited: list[str]
    draft_text: str
    confidence: float

class DisputeState(TypedDict):
    customer_message: str
    tenant_id: str
    transaction_id: str
    evidence: Optional[list[str]]
    fraud_flag: Optional[FraudFlag]
    draft: Optional[DraftResponse]
    critic_verdict: Optional[CriticVerdict]
    approved: Optional[bool]
    submitted: Optional[bool]
    policy_context: Optional[str]

def _normalize(text: str) -> str:
    return text.strip().rstrip(".").lower()

def check_grounding(draft: DraftResponse, evidence: list[str]) -> bool:
    if(len(draft.evidence_cited) == 0) and (len(evidence) > 0):
        return False
    normalized_evidence = [_normalize(e) for e in evidence]
    return all(
        any(_normalize(cited) in ev or ev in _normalize(cited) for ev in normalized_evidence)
        for cited in draft.evidence_cited
    )

def critic_node(state: DisputeState, config: RunnableConfig) -> dict:
    draft = state["draft"]
    evidence = state["evidence"]
    fraud_flag = state["fraud_flag"]
    assert draft is not None and evidence is not None and fraud_flag is not None

    grounded = check_grounding(draft, evidence)
    if not grounded:
        draft = draft_from_evidence(state["customer_message"], evidence, state["policy_context"])  # retry once
        grounded = check_grounding(draft, evidence)

    no_case_evidence = len(evidence) == 0
    escalate = fraud_flag.is_anomaly or not grounded or no_case_evidence

    if no_case_evidence:
        notes = "flagged: no case evidence on file — response relies on external policy context only, needs human review"
    elif fraud_flag.is_anomaly and grounded:
        notes = "flagged: fraud anomaly on this transaction — needs human review"
    elif not grounded:
        notes = "flagged: draft still cites unsupported evidence after retry"
    else:
        notes = "passed: grounded, no fraud signal"

    verdict = CriticVerdict(grounded=grounded, escalate_for_review=escalate, notes=notes)
    log_decision(get_thread_id(config), state["transaction_id"], "critic", {"grounded": verdict.grounded, "escalate_for_review": verdict.escalate_for_review, "notes": verdict.notes})
    return {"draft": draft, "critic_verdict": verdict}

def parse_model_json(text: str) -> str:
    """Claude often wraps JSON in ```json fences; Pydantic needs the raw object."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()
    return text


def message_text(message: Message) -> str:
    for block in message.content:
        if isinstance(block, TextBlock):
            return block.text
    raise TypeError("Claude response had no text block")




# def retrieve_evidence(transaction_id: str) -> list[str]:
#     return [item.text for item in evidence_items if item.transaction_id == transaction_id]

cur = conn.cursor()

def retrieve_evidence_semantic(query_text: str, tenant_id: str, transaction_id: str, top_k: int = 2) -> list[str]:
    query_vector = embedder.encode(query_text)
    cur.execute(
        """
        SELECT text, embedding <=> %s AS distance
        FROM evidence
        WHERE tenant_id = %s AND transaction_id = %s
        ORDER BY distance
        LIMIT %s
        """,
        (query_vector, tenant_id, transaction_id, top_k),
    )
    rows = cur.fetchall()
    return [text for text, distance in rows]

def retrieve_evidence_keyword(query_text: str, tenant_id: str, transaction_id: str, top_k: int = 2) -> list[str]:
    with conn.cursor() as query_cur:
        query_cur.execute(
            """
            SELECT text, ts_rank(text_search, plainto_tsquery('english', %s)) AS rank
            FROM evidence
            WHERE tenant_id = %s AND transaction_id = %s
              AND text_search @@ plainto_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
            """,
            (query_text, tenant_id, transaction_id, query_text, top_k),
        )
        rows = query_cur.fetchall()
    return [text for text, rank in rows]

def retrieve_evidence_hybrid(query_text: str, tenant_id: str, transaction_id: str, top_k: int = 2, rrf_k: int = 60) -> list[str]:
    semantic_results = retrieve_evidence_semantic(query_text, tenant_id, transaction_id, top_k=10)
    keyword_results = retrieve_evidence_keyword(query_text, tenant_id, transaction_id, top_k=10)

    scores: dict[str, float] = {}
    for rank, text in enumerate(semantic_results, start=1):
        scores[text] = scores.get(text, 0.0) + 1.0 / (rrf_k + rank)
    for rank, text in enumerate(keyword_results, start=1):
        scores[text] = scores.get(text, 0.0) + 1.0 / (rrf_k + rank)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [text for text, _ in ranked[:top_k]]

def retrieve_evidence_weaviate(query_text: str, tenant_id: str, transaction_id: str, top_k: int = 2) -> list[str]:
    query_vector = embedder.encode(query_text).tolist()
    evidence_collection = weaviate_client.collections.get("Evidence")
    response = evidence_collection.query.near_vector(
        near_vector=query_vector,
        filters=(
            Filter.by_property("tenant_id").equal(tenant_id)
            & Filter.by_property("transaction_id").equal(transaction_id)
        ),
        limit=top_k,
    )
    texts: list[str] = []
    for obj in response.objects:
        text = obj.properties["text"]
        if isinstance(text, str):
            texts.append(text)
    return texts

def route_after_retrieve(state: DisputeState) -> str:
    evidence = state["evidence"]
    assert evidence is not None, "route_after_retrieve requires retrieve_node to run first"
    if len(evidence) == 0:
        return "policy_lookup"
    return "draft"


def route_after_critic(state: DisputeState) -> str:
    verdict = state["critic_verdict"]
    assert verdict is not None, "route_after_critic requires critic_node to run first"
    return "await_approval" if verdict.escalate_for_review else "submit"

def draft_from_evidence(customer_message: str, evidence: list[str], policy_context: str | None) -> DraftResponse:
    evidence_block = (
        "\n".join(f"- {e}" for e in evidence)
        if evidence
        else "No evidence on file for this transaction — evidence_cited must be an empty list."
    )
    policy_block = (
        f"\n\nRelevant card-network policy (background context, not case-specific evidence):\n{policy_context}"
        if policy_context else ""
    )
    prompt = f"""A customer wrote this dispute message:
"{customer_message}"

Here is the evidence on file for this transaction:
{evidence_block}{policy_block}

Classify it into exactly one of: unrecognized, product_not_received, duplicate, product_unacceptable, and draft a short customer-facing response citing the specific evidence above.
IMPORTANT: evidence_cited must contain exact, verbatim copies of the evidence line(s) you relied on — copy them character-for-character from the evidence list above, do not paraphrase or summarize them there. draft_text may paraphrase naturally for the customer.
You may reference the policy context above in your draft text for framing, but evidence_cited must only ever list items from the evidence section — never from the policy context.
Rate your confidence. Respond with ONLY valid JSON matching this shape:
{{"reason": "...", "evidence_cited": [...], "draft_text": "...", "confidence": 0.0}}"""

    for attempt in range(2):
        try:
            response = client.messages.create(
                model=MODEL_STRONG,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return DraftResponse.model_validate_json(parse_model_json(message_text(response)))
        except ValidationError as e:
            continue

    # If we've tried twice and still can't generate a valid response, route to human review
    return DraftResponse(reason="unrecognized", evidence_cited=[], draft_text="Unable to generate a validated response for this dispute automatically — routed to human review.", confidence=0.0)



def fraud_node(state: DisputeState, config: RunnableConfig) -> dict:
    flag = check_fraud_signals(state["tenant_id"], state["transaction_id"])
    log_decision(get_thread_id(config), state["transaction_id"], "fraud", {"teanant_id": state["tenant_id"],"z_score": flag.z_score, "is_anomaly": flag.is_anomaly, "note": flag.note})
    return {"fraud_flag": flag}

def retrieve_node(state: DisputeState, config: RunnableConfig) -> dict:
    evidence = retrieve_evidence_hybrid(state["customer_message"], state["tenant_id"], state["transaction_id"], top_k=2, rrf_k=60)
    log_decision(get_thread_id(config), state["transaction_id"], "hybrid", {"tenant_id": state["tenant_id"], "transaction_id": state["transaction_id"], "evidence_count": len(evidence), "evidence": evidence})
    return {"evidence": evidence}

def draft_node(state: DisputeState, config: RunnableConfig) -> dict:
    evidence = state["evidence"]
    assert evidence is not None, "draft_node requires retrieve_node to run first"
    draft = draft_from_evidence(state["customer_message"], evidence, state["policy_context"])
    log_decision(get_thread_id(config), state["transaction_id"], "draft",
        {"reason": draft.reason, "confidence": draft.confidence,
         "evidence_cited": draft.evidence_cited, "draft_text": draft.draft_text,
         "used_policy_context": state["policy_context"] is not None},)
    return {"draft": draft}

def await_approval_node(state: DisputeState, config: RunnableConfig) -> dict:
    assert state["draft"] is not None and state["critic_verdict"] is not None
    decision = interrupt({
        "action": "approve_dispute_response",
        "draft_text": state["draft"].draft_text,
        "critic_notes": state["critic_verdict"].notes,
        "escalate_for_review": state["critic_verdict"].escalate_for_review,
    })
    approved = bool(decision.get("approved", False))
    log_decision(get_thread_id(config),state["transaction_id"], "await_approval",{"approved": approved},)
    return {"approved": approved}

def submit_node(state: DisputeState, config: RunnableConfig) -> dict:
    verdict = state["critic_verdict"]
    assert verdict is not None, "submit_node requires critic_node to run first"

    if verdict.escalate_for_review:
        submitted = bool(state["approved"])
        approval_source = "human" if submitted else "human_rejected"
    else:
        submitted = True
        approval_source = "auto"

    print(f"[SUBMITTED] response for {state['transaction_id']} ({approval_source})" if submitted
          else f"[REJECTED] response for {state['transaction_id']} was not approved")
    log_decision(
        get_thread_id(config), state["transaction_id"], "submit",
        {"approved": state["approved"], "submitted": submitted, "approval_source": approval_source},
    )
    return {"submitted": submitted}

REASON_TO_SEARCH_PHRASE = {
    "unrecognized": "unauthorized transaction chargeback reason code",
    "product_not_received": "merchandise or service not received chargeback reason code",
    "duplicate": "duplicate processing chargeback reason code",
    "product_unacceptable": "defective or not as described chargeback reason code",
}


def classify_dispute_reason(customer_message: str) -> str:
    prompt = f"""A customer wrote this dispute message:
                "{customer_message}"

    Classify it into exactly one of: unrecognized, product_not_received, duplicate, product_unacceptable.
    Respond with ONLY the category name, nothing else."""

    response = client.messages.create(
        model=MODEL_CHEAP,
        max_tokens=20,
        messages=[{"role": "user", "content": prompt}],
    )
    return message_text(response).strip()


def format_policy_context(results: list[dict]) -> str:
    """Turn Tavily's raw results into a citable block for the drafting prompt."""
    lines = []
    for r in results[:3]:  # top 3 by relevance score
        lines.append(f"- {r['title']} ({r['url']}): {r['content']}")
    return "\n".join(lines)


def policy_lookup_node(state: DisputeState, config: RunnableConfig) -> dict:
    reason = classify_dispute_reason(state["customer_message"])
    search_phrase = REASON_TO_SEARCH_PHRASE.get(reason, "chargeback dispute policy")
    query = f"Visa Mastercard {search_phrase}"
    response = tavily_client.search(query=query, max_results=3)
    policy_context = format_policy_context(response["results"])
    log_decision( get_thread_id(config), state["transaction_id"], "policy_lookup", {"query": query, "num_results": len(response["results"])})
    return {"policy_context": policy_context}

graph = StateGraph(DisputeState)
graph.add_node("retrieve",retrieve_node)
graph.add_node("fraud", fraud_node)
graph.add_node("draft", draft_node, defer=True)  # wait for whichever upstream branch actually ran
graph.add_node("critic", critic_node)
graph.add_node("await_approval", await_approval_node)
graph.add_node("submit", submit_node)
graph.add_node("policy_lookup", policy_lookup_node)

graph.add_edge(START, "fraud")
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "draft")
graph.add_edge("fraud", "draft")
graph.add_edge("draft", "critic")
graph.add_edge("policy_lookup", "draft")

graph.add_conditional_edges("critic", route_after_critic, {
    "await_approval": "await_approval",
    "submit": "submit",
})

graph.add_conditional_edges(
    "retrieve",
    route_after_retrieve,
    {"policy_lookup": "policy_lookup", "draft": "draft"},
)

graph.add_edge("await_approval", "submit")
graph.add_edge("submit", END)


_checkpointer_cm = PostgresSaver.from_conn_string(DB_URI)
checkpointer = _checkpointer_cm.__enter__()  # entered manually, never exited — see atexit below
checkpointer.setup()  # creates the checkpoint tables if they don't already exist
atexit.register(lambda: _checkpointer_cm.__exit__(None, None, None))

app_graph = graph.compile(checkpointer=checkpointer)


class DisputeRequest(BaseModel):
    tenant_id: str
    transaction_id: str
    customer_message: str


class ApprovalRequest(BaseModel):
    approved: bool


@app.post("/disputes")
def submit_dispute(req: DisputeRequest):
    if get_transaction(req.tenant_id, req.transaction_id) is None:
        raise HTTPException(status_code=404, detail="transaction not found")

    thread_id = f"dispute-{req.tenant_id}-{req.transaction_id}"
    config = make_config(thread_id)
    result = app_graph.invoke({
        "customer_message": req.customer_message,
        "tenant_id": req.tenant_id,
        "transaction_id": req.transaction_id,
        "evidence": None,
        "fraud_flag": None,
        "draft": None,
        "critic_verdict": None,
        "approved": None,
        "submitted": None,
        "policy_context": None,
    }, config=config)

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        return {
            "status": "awaiting_approval",
            "thread_id": thread_id,
            "draft_text": payload["draft_text"],
            "critic_notes": payload["critic_notes"],
        }

    return {
        "status": "submitted" if result["submitted"] else "rejected",
        "thread_id": thread_id,
        "draft_text": result["draft"].draft_text,
    }


@app.get("/", response_class=HTMLResponse)
def frontend():
    with open("static/index.html") as f:
        return f.read()

@app.post("/disputes/{thread_id}/approve")
def approve_dispute_route(thread_id: str, req: ApprovalRequest):
    config = make_config(thread_id)
    result = app_graph.invoke(Command(resume={"approved": req.approved}), config=config)
    return {
        "status": "submitted" if result["submitted"] else "rejected",
        "thread_id": thread_id,
        "draft_text": result["draft"].draft_text,
    }

# graph2 = StateGraph(PolicyState)
# graph2.add_node("retrieve_policy", retrieve_policy_node)
# graph2.add_edge(START, "retrieve_policy")
# graph2.add_edge("retrieve_policy", END)

# checkpointer2 = MemorySaver()
# app_graph2 = graph2.compile(checkpointer=checkpointer2)

# @app.post("/classify")
# def classify_endpoint(request: ClassifyRequest) -> DraftClassification:
#     return classify_dispute(request.customer_message)




# def recommend_action(dispute: Dispute, transaction: Transaction) -> str:
#     if dispute.reason == "product_not_received":
#         if transaction.delivered and transaction.delivery_address_matches_billing:
#             return "contest — delivery confirmed to matching address"
#         return "accept — no solid delivery proof"

#     if dispute.reason == "duplicate":
#         return "contest — check for a matching second transaction"

#     if dispute.reason == "unrecognized":
#         if transaction.delivered:
#             return "contest — delivery evidence exists"
#         return "needs_review — no strong evidence either way"

#     return "needs_review — unhandled reason code"




# disputes = [
#     Dispute(id="D1", transaction_id="T1", reason="unrecognized"),      # this is Priya's case
#     Dispute(id="D2", transaction_id="T2", reason="product_not_received"),
#     Dispute(id="D3", transaction_id="T3", reason="duplicate"),
#     Dispute(id="D4", transaction_id="T4", reason="product_unacceptable"),
# ]




# def classify_dispute(customer_message: str) -> DraftClassification:
#     prompt = f"""A customer wrote this message about a payment dispute:

# "{customer_message}"

# Classify it into exactly one of: unrecognized, product_not_received, duplicate, product_unacceptable.
# Respond with ONLY valid JSON matching this shape, nothing else:
# {{"reason": "...", "confidence": 0.0 to 1.0, "explanation": "one sentence"}}"""

#     response = client.messages.create(
#         model="claude-sonnet-4-6",
#         max_tokens=1024,
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return DraftClassification.model_validate_json(parse_model_json(message_text(response)))


def run_dispute(customer_message: str, tenant_id: str, transaction_id: str, thread_id: str) -> dict:
    config = make_config(thread_id)
    result = app_graph.invoke({
        "customer_message": customer_message,
        "tenant_id": tenant_id,
        "transaction_id": transaction_id,
        "evidence": None,
        "fraud_flag": None,
        "draft": None,
        "critic_verdict": None,
        "approved": None,
        "submitted": None,
        "policy_context": None,
    }, config=config)

    if "__interrupt__" in result:
        print("PAUSED —", result)
        result = app_graph.invoke(Command(resume={"approved": True}), config=config)
        print("RESUMED —", result)
    else:
        print("COMPLETED (no pause needed) —", result)

    return result

def benchmark(fn, *args, n=20, **kwargs):
    fn(*args, **kwargs)  # warm-up call, not counted — excludes first-call connection setup
    times = []
    for _ in range(n):
        start = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - start)
    times.sort()
    return {
        "mean_ms": round(sum(times) / len(times) * 1000, 2),
        "median_ms": round(times[len(times) // 2] * 1000, 2),
        "min_ms": round(times[0] * 1000, 2),
        "max_ms": round(times[-1] * 1000, 2),
    }



def main():
    # tx_by_id = {t.id: t for t in transactions}
    # for d in disputes:
    #     tx = tx_by_id[d.transaction_id]
    #     action = recommend_action(d, tx)
    #     print(f"{d.id} (reason: {d.reason}) on {tx.id} (₹{tx.amount}) -> {action}")

    # print(draft_response("I don't recognize this charge on my card, I never ordered anything.", "T1"))


    # for item in evidence_items:
    #     vector = embedder.encode(item.text)
    #     cur.execute(
    #         "INSERT INTO evidence (transaction_id, text, embedding) VALUES (%s, %s, %s)",
    #         (item.transaction_id, item.text, vector),
    #     )
    # conn.commit()

    # print(retrieve_evidence_semantic("the customer says their package never arrived"))

    # run_dispute(
    # "I don't recognize this charge on my card, I never ordered anything.",
    # "electromart",
    # "T1",
    # "dispute-T1-regression-check",
    # )
    run_dispute("what about tracking TRK556", "subscribebox", "TRK556", "dispute-TRK556-demo")
    # print("hybrid:", retrieve_evidence_hybrid("what about tracking TRK556", "electromart", "T1", top_k=2))
    # run_dispute("I have a duplicate charge on my card, I never ordered anything.","electromart", "T2", "dispute-T2-demo")
    # print("pgvector:", retrieve_evidence_semantic("what about tracking TRK556", "electromart", "T1", top_k=2))
    # print("weaviate:", retrieve_evidence_weaviate("what about tracking TRK556", "electromart", "T1", top_k=2))
    # print("pgvector:", benchmark(retrieve_evidence_semantic, "what about tracking TRK556", "electromart", "T1", top_k=2));
    # print("weaviate:", benchmark(retrieve_evidence_weaviate, "what about tracking TRK556", "electromart", "T1", top_k=2));

if __name__ == "__main__":
    main()