import json
from disputedesk import app_graph

GOLDEN_CASES = [
    {
        "id": "T1-electromart-real-evidence-autosubmit",
        "customer_message": "I don't recognize this charge on my card, I never ordered anything.",
        "tenant_id": "electromart",
        "transaction_id": "T1",
        # Verified live earlier in this project: real evidence, z=1.0 (non-anomalous),
        # verbatim-grounded draft, auto-submitted with no HITL pause.
        "expected": {"evidence_empty": False, "is_anomaly": False, "escalate_for_review": False, "reason": "unrecognized"},
    },
    {
        "id": "cross-tenant-T1-as-subscribebox",
        "customer_message": "I never received this order.",
        "tenant_id": "subscribebox",  # wrong tenant for T1
        "transaction_id": "T1",
        # Verified: wrong-tenant lookup returns zero evidence -> policy_lookup fires ->
        # no_case_evidence forces escalation regardless of what the model does. Structural
        # guarantee, not model-dependent, so this is a strong case to have in the suite.
        "expected": {"evidence_empty": True, "is_anomaly": False, "escalate_for_review": True, "reason": None},
    },
    {
        "id": "unknown-transaction-policy-fallback",
        "customer_message": "what about tracking TRK556",
        "tenant_id": "subscribebox",
        "transaction_id": "TRK556",  # doesn't exist as a transaction at all
        # Verified live, twice, earlier in this session.
        "expected": {"evidence_empty": True, "is_anomaly": False, "escalate_for_review": True, "reason": "product_not_received"},
    },
    {
        "id": "T3-subscribebox-real-evidence",
        "customer_message": "I was charged twice for this order, please refund the duplicate.",
        "tenant_id": "subscribebox",
        "transaction_id": "T3",
        # NOT yet run in this conversation — this is a hypothesis, not a verified fact.
        # z-score for T3 is -1.0 (non-anomalous, same 2-point-population math as T1/T2),
        # and evidence explicitly says "no duplicate found" — so I'd expect a grounded,
        # non-escalating draft that tells the customer no duplicate exists. But confirm
        # `reason` and `escalate_for_review` from the real first run and correct this
        # entry rather than assuming I got it right.
        "expected": {"evidence_empty": False, "is_anomaly": False, "escalate_for_review": False, "reason": "duplicate"},
    },
]

def run_case(case: dict) -> dict:
    config = {"configurable": {"thread_id": f"eval-{case['id']}"}}
    result = app_graph.invoke(
        {
            "customer_message": case["customer_message"],
            "tenant_id": case["tenant_id"],
            "transaction_id": case["transaction_id"],
            "evidence": None, "fraud_flag": None, "draft": None,
            "critic_verdict": None, "approved": None, "submitted": None,
            "policy_context": None,
        },
        config=config,
    )
    evidence = result.get("evidence") or []
    fraud_flag = result.get("fraud_flag")
    draft = result.get("draft")
    critic_verdict = result.get("critic_verdict")
    return {
        "evidence_empty": len(evidence) == 0,
        "is_anomaly": fraud_flag.is_anomaly if fraud_flag else None,
        "reason": draft.reason if draft else None,
        "escalate_for_review": critic_verdict.escalate_for_review if critic_verdict else None,
    }


def score_case(case: dict, actual: dict) -> dict:
    expected = case["expected"]
    hard_fields = ("evidence_empty", "is_anomaly", "escalate_for_review")
    hard_mismatches = [
        f"{f}: expected {expected[f]}, got {actual[f]}"
        for f in hard_fields if actual[f] != expected[f]
    ]
    soft_mismatches = []
    if expected.get("reason") is not None and actual["reason"] != expected["reason"]:
        soft_mismatches.append(f"reason: expected {expected['reason']}, got {actual['reason']}")
    return {"passed": not hard_mismatches, "hard_mismatches": hard_mismatches, "soft_mismatches": soft_mismatches}


def main():
    results = []
    for case in GOLDEN_CASES:
        actual = run_case(case)
        verdict = score_case(case, actual)
        results.append({"id": case["id"], "actual": actual, **verdict})

        print(f"[{'PASS' if verdict['passed'] else 'FAIL'}] {case['id']}")
        for m in verdict["hard_mismatches"]:
            print(f"    HARD: {m}")
        for m in verdict["soft_mismatches"]:
            print(f"    warn: {m}")

    passed = sum(r["passed"] for r in results)
    print(f"\n{passed}/{len(results)} golden cases passed (hard checks)")

    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    if passed < len(results):
        raise SystemExit(1)

    

if __name__ == "__main__":
    main()