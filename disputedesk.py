#!/usr/bin/env python3
from dataclasses import dataclass
from datetime import date
from typing import Literal
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
from anthropic_api_call import client
from fastapi import FastAPI
app = FastAPI()

embedder = SentenceTransformer("all-MiniLM-L6-v2")

conn = psycopg2.connect(
    host="localhost", port=5432, dbname="postgres", user="postgres", password="devpassword"
)
register_vector(conn)  # teaches psycopg2 how to send Python vectors to Postgres's vector column

def parse_model_json(text: str) -> str:
    """Claude often wraps JSON in ```json fences; Pydantic needs the raw object."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: text.rfind("```")]
        text = text.strip()
    return text


class DraftClassification(BaseModel):
    reason: Literal["unrecognized", "product_not_received", "duplicate", "product_unacceptable"]
    confidence: float
    explanation: str

class ClassifyRequest(BaseModel):
    customer_message: str

@dataclass
class EvidenceItem:
    transaction_id: str
    text: str

evidence_items = [
    EvidenceItem("T1", "Courier tracking TRK556 shows delivered Aug 9, signed at the billing address."),
    EvidenceItem("T2", "No delivery confirmation on file for this order."),
    EvidenceItem("T3", "Order history shows only one charge of ₹50 on this date — no duplicate found."),
]

class DraftResponse(BaseModel):
    reason: Literal["unrecognized", "product_not_received", "duplicate", "product_unacceptable"]
    evidence_cited: list[str]
    draft_text: str
    confidence: float


def retrieve_evidence(transaction_id: str) -> list[str]:
    return [item.text for item in evidence_items if item.transaction_id == transaction_id]

cur = conn.cursor()

def retrieve_evidence_semantic(query_text: str, transaction_id: str, top_k: int = 2) -> list[str]:
    query_vector = embedder.encode(query_text)
    cur.execute(
        """
        SELECT text, embedding <=> %s AS distance
        FROM evidence
        WHERE transaction_id = %s
        ORDER BY distance
        LIMIT %s
        """,
        (query_vector, transaction_id, top_k),
    )
    rows = cur.fetchall()
    return [text for text, distance in rows]

def draft_response(customer_message: str, transaction_id: str) -> DraftResponse:
    evidence = retrieve_evidence_semantic(customer_message, transaction_id, top_k=2)
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
    return DraftResponse.model_validate_json(parse_model_json(response.content[0].text))

@app.post("/classify")
def classify_endpoint(request: ClassifyRequest) -> DraftClassification:
    return classify_dispute(request.customer_message)

@dataclass
class Transaction:
    id: str
    amount: float
    order_date: date
    delivered: bool
    delivery_address_matches_billing: bool


class Dispute(BaseModel):
    id: str
    transaction_id: str
    reason: Literal["unrecognized", "product_not_received", "duplicate", "product_unacceptable"]


def recommend_action(dispute: Dispute, transaction: Transaction) -> str:
    if dispute.reason == "product_not_received":
        if transaction.delivered and transaction.delivery_address_matches_billing:
            return "contest — delivery confirmed to matching address"
        return "accept — no solid delivery proof"

    if dispute.reason == "duplicate":
        return "contest — check for a matching second transaction"

    if dispute.reason == "unrecognized":
        if transaction.delivered:
            return "contest — delivery evidence exists"
        return "needs_review — no strong evidence either way"

    return "needs_review — unhandled reason code"


# a tiny pretend database, standing in for real records
transactions = [
    Transaction("T1", 2499, date(2026, 8, 5), delivered=True, delivery_address_matches_billing=True),
    Transaction("T2", 1200, date(2026, 8, 10), delivered=False, delivery_address_matches_billing=False),
    Transaction("T3", 50, date(2026, 8, 12), delivered=True, delivery_address_matches_billing=True),
    Transaction("T4", 800, date(2026, 8, 15), delivered=True, delivery_address_matches_billing=True),
    ]

disputes = [
    Dispute(id="D1", transaction_id="T1", reason="unrecognized"),      # this is Priya's case
    Dispute(id="D2", transaction_id="T2", reason="product_not_received"),
    Dispute(id="D3", transaction_id="T3", reason="duplicate"),
    Dispute(id="D4", transaction_id="T4", reason="product_unacceptable"),
]




def classify_dispute(customer_message: str) -> DraftClassification:
    prompt = f"""A customer wrote this message about a payment dispute:

"{customer_message}"

Classify it into exactly one of: unrecognized, product_not_received, duplicate, product_unacceptable.
Respond with ONLY valid JSON matching this shape, nothing else:
{{"reason": "...", "confidence": 0.0 to 1.0, "explanation": "one sentence"}}"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return DraftClassification.model_validate_json(parse_model_json(response.content[0].text))

def main():
    # tx_by_id = {t.id: t for t in transactions}
    # for d in disputes:
    #     tx = tx_by_id[d.transaction_id]
    #     action = recommend_action(d, tx)
    #     print(f"{d.id} (reason: {d.reason}) on {tx.id} (₹{tx.amount}) -> {action}")

    print(draft_response("I don't recognize this charge on my card, I never ordered anything.", "T1"))


    # for item in evidence_items:
    #     vector = embedder.encode(item.text)
    #     cur.execute(
    #         "INSERT INTO evidence (transaction_id, text, embedding) VALUES (%s, %s, %s)",
    #         (item.transaction_id, item.text, vector),
    #     )
    # conn.commit()

    # print(retrieve_evidence_semantic("the customer says their package never arrived"))



if __name__ == "__main__":
    main()