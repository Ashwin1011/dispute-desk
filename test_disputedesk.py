from datetime import date
from disputedesk import Transaction, Dispute, recommend_action


def test_product_not_received_with_delivery_proof_is_contested():
    tx = Transaction("T1", 2499, date(2026, 8, 5), delivered=True, delivery_address_matches_billing=True)
    d = Dispute("D1", "T1", "product_not_received")
    assert recommend_action(d, tx) == "contest — delivery confirmed to matching address"


def test_product_not_received_without_delivery_proof_is_accepted():
    tx = Transaction("T2", 1200, date(2026, 8, 10), delivered=False, delivery_address_matches_billing=False)
    d = Dispute("D2", "T2", "product_not_received")
    assert recommend_action(d, tx) == "accept — no solid delivery proof"


def test_duplicate_always_recommends_contest():
    tx = Transaction("T3", 50, date(2026, 8, 12), delivered=True, delivery_address_matches_billing=True)
    d = Dispute("D3", "T3", "duplicate")
    assert recommend_action(d, tx) == "contest — check for a matching second transaction"


def test_unhandled_reason_falls_back_to_needs_review():
    tx = Transaction("T4", 800, date(2026, 8, 15), delivered=True, delivery_address_matches_billing=True)
    d = Dispute("D4", "T4", "product_unacceptable")
    assert recommend_action(d, tx) == "needs_review — unhandled reason code"