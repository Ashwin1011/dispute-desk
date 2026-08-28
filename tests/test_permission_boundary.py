import pytest
from disputedesk import (
    retrieve_evidence_semantic, retrieve_evidence_keyword, retrieve_evidence_hybrid,
    check_fraud_signals, transactions, evidence_items,
    retrieve_evidence_weaviate,
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


def test_fraud_check_works_for_correct_tenant():
    # Positive path for check_fraud_signals — every existing fraud test only exercises
    # wrong-tenant lookups, which return early before statistics.mean/pstdev ever run.
    # That left the real z-score computation completely uncovered (a commented-out
    # `import statistics` would have shipped silently). This exercises it directly,
    # on a known non-anomalous transaction, and pins the exact z-score so a broken
    # computation — not just a missing import — also fails loudly.
    flag = check_fraud_signals("electromart", "T1")
    assert flag.note != "transaction not found for this tenant"
    assert isinstance(flag.z_score, float)
    assert flag.z_score == pytest.approx(1.0)
    assert flag.is_anomaly is False


def test_fraud_check_flags_real_anomaly():
    # Companion case for the other branch: gizmohub/T8 is a genuine 4-transaction
    # population with one real outlier, so this is the one live case where
    # is_anomaly is actually expected to fire end to end.
    flag = check_fraud_signals("gizmohub", "T8")
    assert flag.note != "transaction not found for this tenant"
    assert flag.is_anomaly is True
    assert flag.z_score > 1.5

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

def test_retrieve_evidence_weaviate_works_for_correct_tenant():
    evidence = retrieve_evidence_weaviate("tracking TRK556", "electromart", "T1", top_k=2)
    assert len(evidence) > 0


@pytest.mark.parametrize("transaction_id,wrong_tenant_id", list(wrong_tenant_pairs()))
def test_retrieve_evidence_weaviate_blocks_cross_tenant_access(transaction_id, wrong_tenant_id):
    query = first_evidence_text(transaction_id)
    evidence = retrieve_evidence_weaviate(query, wrong_tenant_id, transaction_id, top_k=5)
    assert evidence == [], (
        f"tenant '{wrong_tenant_id}' retrieved evidence for transaction '{transaction_id}' via weaviate"
    )