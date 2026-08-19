import json

def classify_dispute(amount, has_delivery_confirmation):
    if amount > 5000 and not has_delivery_confirmation:
        return "likely_product_not_received"
    if amount < 100:
        return "likely_duplicate_charge"
    return "needs_review"

def main():
    print("DisputeDesk v0 - enter amount and y/n for delivery confirmation")
    while True:
        raw = input("> ")
        if raw.strip().lower() == "quit":
            break
        amount_str, has_conf = raw.split()
        result = classify_dispute(float(amount_str), has_conf.lower() == "y")
        print(f"  classification: {result}")

if __name__ == "__main__":
    main()
