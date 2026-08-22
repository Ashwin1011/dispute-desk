from disputedesk import conn, embedder, evidence_items

cur = conn.cursor()
for item in evidence_items:
    vector = embedder.encode(item.text)
    cur.execute(
        "INSERT INTO evidence (tenant_id, transaction_id, text, embedding) VALUES (%s, %s, %s, %s)",
        (item.tenant_id, item.transaction_id, item.text, vector),
    )
conn.commit()
print(f"seeded {len(evidence_items)} evidence rows")