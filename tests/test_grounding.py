from disputedesk import check_grounding, DraftResponse

REAL_EVIDENCE = [
    "Courier tracking TRK556 shows delivered Aug 9, signed at the billing address.",
    "Order confirmation email was sent to the customer on Aug 3 with estimated delivery Aug 9-11.",
]


def make_draft(evidence_cited: list[str]) -> DraftResponse:
    return DraftResponse(
        reason="unrecognized",
        evidence_cited=evidence_cited,
        draft_text="placeholder",
        confidence=0.5,
    )

# Should always return False
def test_check_grounding_returns_false_for_empty_citation_with_real_evidence():
    draft = make_draft(evidence_cited=[])
    assert check_grounding(draft, REAL_EVIDENCE) is False

# Should return True
def test_check_grounding_returns_true_for_a_real_matching_citation():
    draft = make_draft(evidence_cited=[REAL_EVIDENCE[0]])
    assert check_grounding(draft, REAL_EVIDENCE) is True

# Should return True
def test_check_grounding_vacuously_true_with_no_evidence_and_no_citations():
    draft = make_draft(evidence_cited=[])
    assert check_grounding(draft, []) is True