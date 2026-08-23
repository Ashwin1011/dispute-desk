import pytest
from disputedesk import (
    retrieve_evidence_semantic, retrieve_evidence_keyword, retrieve_evidence_hybrid,
    check_fraud_signals, transactions, evidence_items,
)

ALL_TENANTS = sorted({t.tenant_id for t in transactions})


def wrong_tenant_pairs():
    """Every (transaction, tenant) pair where that tenant does NOT own the transaction."""
    for tx in transactions:
        for tenant_id in ALL_TENANTS:
            if tenant_id != tx.tenant_id:
                yield tx.id, tenant_id


@pytest.mark.parametrize("transaction_id,wrong_tenant_id", list(wrong_tenant_pairs()))
def test_retrieve_evidence_blocks_cross_tenant_access(transaction_id, wrong_tenant_id):
    evidence = retrieve_evidence_semantic("any query text", wrong_tenant_id, transaction_id, top_k=5)
    assert evidence == [], (
        f"tenant '{wrong_tenant_id}' retrieved evidence for transaction '{transaction_id}', "
        f"which it does not own — cross-tenant leakage"
    )


@pytest.mark.parametrize("transaction_id,wrong_tenant_id", list(wrong_tenant_pairs()))
def test_fraud_check_blocks_cross_tenant_access(transaction_id, wrong_tenant_id):
    flag = check_fraud_signals(wrong_tenant_id, transaction_id)
    assert flag.note == "transaction not found for this tenant", (
        f"tenant '{wrong_tenant_id}' got a real fraud signal for transaction '{transaction_id}', "
        f"which it does not own"
    )


def test_retrieve_evidence_works_for_correct_tenant():
    # sanity check — the boundary should block wrong tenants, not block everyone
    evidence = retrieve_evidence_semantic("delivery status", "electromart", "T1", top_k=2)
    assert len(evidence) > 0

def first_evidence_text(transaction_id: str) -> str:
    for item in evidence_items:
        if item.transaction_id == transaction_id:
            return item.text
    raise AssertionError(
        f"no evidence fixture exists for transaction '{transaction_id}' — "
        f"add one to evidence_items before testing its permission boundary"
    )


@pytest.mark.parametrize("transaction_id,wrong_tenant_id", list(wrong_tenant_pairs()))
def test_retrieve_evidence_keyword_blocks_cross_tenant_access(transaction_id, wrong_tenant_id):
    query = first_evidence_text(transaction_id)  # guaranteed to keyword-match if the boundary were broken
    evidence = retrieve_evidence_keyword(query, wrong_tenant_id, transaction_id, top_k=5)
    assert evidence == [], (
        f"tenant '{wrong_tenant_id}' retrieved evidence for transaction '{transaction_id}' via keyword search"
    )


@pytest.mark.parametrize("transaction_id,wrong_tenant_id", list(wrong_tenant_pairs()))
def test_retrieve_evidence_hybrid_blocks_cross_tenant_access(transaction_id, wrong_tenant_id):
    query = first_evidence_text(transaction_id)
    evidence = retrieve_evidence_hybrid(query, wrong_tenant_id, transaction_id, top_k=5)
    assert evidence == [], (
        f"tenant '{wrong_tenant_id}' retrieved evidence for transaction '{transaction_id}' via hybrid search"
    )


def test_retrieve_evidence_keyword_works_for_correct_tenant():
    evidence = retrieve_evidence_keyword("tracking TRK556", "electromart", "T1", top_k=2)
    assert len(evidence) > 0


def test_retrieve_evidence_hybrid_works_for_correct_tenant():
    evidence = retrieve_evidence_hybrid("tracking TRK556", "electromart", "T1", top_k=2)
    assert len(evidence) > 0