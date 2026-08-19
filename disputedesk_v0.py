from dataclasses import dataclass

@dataclass
class Dispute:
    amount: float
    reason: str

def classify_dispute(dispute: Dispute) -> str:
    if dispute.amount > 5000 and dispute.reason == "product_not_received":
        return "likely_product_not_received"
    elif dispute.amount > 5000 and dispute.reason == "duplicate":
        return "likely_duplicate_charge"
    elif dispute.amount > 5000 and dispute.reason == "unrecognized":
        return "likely_unrecognized_charge"
    elif dispute.amount < 100 and dispute.reason == "product_not_received":
        return "likely_product_not_received"
    elif dispute.amount < 100 and dispute.reason == "duplicate":
        return "likely_duplicate_charge"
    elif dispute.amount < 100 and dispute.reason == "unrecognized":
        return "likely_unrecognized_charge"
    else:
        return "needs_review"
def main():
    print("DisputeDesk v0.5 — enter amount and reason for dispute")
    while True:
        raw = input("> ")
        if raw.strip().lower() == "quit":
            break
        amount_str, reason_str = raw.split()
        dispute = Dispute(
            amount=float(amount_str),
            reason=reason_str
        )
        result = classify_dispute(dispute)
        print(f"  classification: {result}")

if __name__ == "__main__":
    main()
