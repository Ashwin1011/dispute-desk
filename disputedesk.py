#!/usr/bin/env python
from dataclasses import dataclass
from datetime import date
from typing import Literal, TypedDict, Optional
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt, Command
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
from anthropic_api_call import client
from anthropic.types import Message, TextBlock
from fastapi import FastAPI
import statistics
import json
import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter
import time

weaviate_client = weaviate.connect_to_local()
app = FastAPI()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

conn = psycopg2.connect(
    host="localhost", port=5432, dbname="postgres", user="postgres", password="devpassword"
)
register_vector(conn)  # teaches psycopg2 how to send Python vectors to Postgres's vector column


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
    EvidenceItem("subscribebox", "T4", "Delivery confirmed via courier tracking SBX882, signed for at the customer's registered address on Aug 15.")
]

# a tiny pretend database, standing in for real records
transactions = [
    Transaction("T1","electromart", 2499, date(2026, 8, 5), delivered=True, delivery_address_matches_billing=True),
    Transaction("T2","electromart", 1200, date(2026, 8, 10), delivered=False, delivery_address_matches_billing=False),
    Transaction("T3","subscribebox", 50, date(2026, 8, 12), delivered=True, delivery_address_matches_billing=True),
    Transaction("T4","subscribebox", 800, date(2026, 8, 15), delivered=True, delivery_address_matches_billing=True),
    ]

import statistics

class FraudFlag(BaseModel):
    z_score: float
    is_anomaly: bool
    note: str

def check_fraud_signals(tenant_id: str, transaction_id: str) -> FraudFlag:
    tenant_txs = [t for t in transactions if t.tenant_id == tenant_id]
    tx = next((t for t in tenant_txs if t.id == transaction_id), None)
    if tx is None:
        return FraudFlag(z_score=0.0, is_anomaly=False, note="transaction not found for this tenant")

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

def _normalize(text: str) -> str:
    return text.strip().rstrip(".").lower()

def check_grounding(draft: DraftResponse, evidence: list[str]) -> bool:
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
        draft = draft_from_evidence(state["customer_message"], evidence)  # retry once
        grounded = check_grounding(draft, evidence)

    escalate = fraud_flag.is_anomaly or not grounded
    if fraud_flag.is_anomaly and grounded:
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


def route_after_critic(state: DisputeState) -> str:
    verdict = state["critic_verdict"]
    assert verdict is not None, "route_after_critic requires critic_node to run first"
    return "await_approval" if verdict.escalate_for_review else "submit"

def draft_from_evidence(customer_message: str, evidence: list[str]) -> DraftResponse:
    evidence_block = "\n".join(f"- {e}" for e in evidence) or "No evidence on file."
    prompt = f"""A customer wrote this dispute message:
"{customer_message}"
Here is the evidence on file for this transaction:
{evidence_block}
Classify it into exactly one of: unrecognized, product_not_received, duplicate, product_unacceptable, draft a short response citing the specific evidence above,
and rate your confidence. Respond with ONLY valid JSON matching this shape:
{{"reason": "...", "evidence_cited": ["..."], "draft_text": "...", "confidence": 0.0}}"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    return DraftResponse.model_validate_json(parse_model_json(message_text(response)))


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
    draft = draft_from_evidence(state["customer_message"], evidence)
    log_decision(get_thread_id(config), state["transaction_id"], "draft",
        {"reason": draft.reason, "confidence": draft.confidence,
         "evidence_cited": draft.evidence_cited, "draft_text": draft.draft_text},)
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

graph = StateGraph(DisputeState)
graph.add_node("retrieve",retrieve_node)
graph.add_node("fraud", fraud_node)
graph.add_node("draft", draft_node)
graph.add_node("critic", critic_node)
graph.add_node("await_approval", await_approval_node)
graph.add_node("submit", submit_node)

graph.add_edge(START, "fraud")
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "draft")
graph.add_edge("fraud", "draft")
graph.add_edge("draft", "critic")

graph.add_conditional_edges("critic", route_after_critic, {
    "await_approval": "await_approval",
    "submit": "submit",
})
graph.add_edge("await_approval", "submit")
graph.add_edge("submit", END)

checkpointer = MemorySaver()
app_graph = graph.compile(checkpointer=checkpointer)

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
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
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

    # run_dispute("I have a duplicate charge on my card, I never ordered anything.", "electromart", "T1", "dispute-T1-demo")
    # print("hybrid:", retrieve_evidence_hybrid("what about tracking TRK556", "electromart", "T1", top_k=2))
    # run_dispute("I have a duplicate charge on my card, I never ordered anything.", "T2", "dispute-T2-demo")
    # print("pgvector:", retrieve_evidence_semantic("what about tracking TRK556", "electromart", "T1", top_k=2))
    # print("weaviate:", retrieve_evidence_weaviate("what about tracking TRK556", "electromart", "T1", top_k=2))
    print("pgvector:", benchmark(retrieve_evidence_semantic, "what about tracking TRK556", "electromart", "T1", top_k=2));
    print("weaviate:", benchmark(retrieve_evidence_weaviate, "what about tracking TRK556", "electromart", "T1", top_k=2));
    weaviate_client.close()
    conn.close()

if __name__ == "__main__":
    main()