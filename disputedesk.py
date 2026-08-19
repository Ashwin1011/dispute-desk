#!/usr/bin/env python3
from dataclasses import dataclass
from datetime import date
from typing import Literal
from pydantic import BaseModel


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


def main():
    tx_by_id = {t.id: t for t in transactions}
    for d in disputes:
        tx = tx_by_id[d.transaction_id]
        action = recommend_action(d, tx)
        print(f"{d.id} (reason: {d.reason}) on {tx.id} (₹{tx.amount}) -> {action}")


if __name__ == "__main__":
    main()