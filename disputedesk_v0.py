from dataclasses import dataclass
from datetime import date


@dataclass
class Transaction:
    id: str
    amount: float
    order_date: date
    delivered: bool
    delivery_address_matches_billing: bool


@dataclass
class Dispute:
    id: str
    transaction_id: str
    reason: str  # "unrecognized", "product_not_received", "duplicate", "product_unacceptable"


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
]

disputes = [
    Dispute("D1", "T1", "unrecognized"),      # this is Priya's case
    Dispute("D2", "T2", "product_not_received"),
    Dispute("D3", "T3", "duplicate"),
]


def main():
    tx_by_id = {t.id: t for t in transactions}
    for d in disputes:
        tx = tx_by_id[d.transaction_id]
        action = recommend_action(d, tx)
        print(f"{d.id} (reason: {d.reason}) on {tx.id} (₹{tx.amount}) -> {action}")


if __name__ == "__main__":
    main()