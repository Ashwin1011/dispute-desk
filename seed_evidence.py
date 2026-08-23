from weaviate.classes.config import Configure, DataType, Property
from disputedesk import conn, embedder, evidence_items, weaviate_client

# --- Postgres ---
cur = conn.cursor()
for item in evidence_items:
    vector = embedder.encode(item.text)
    cur.execute(
        "INSERT INTO evidence (tenant_id, transaction_id, text, embedding) VALUES (%s, %s, %s, %s)",
        (item.tenant_id, item.transaction_id, item.text, vector),
    )
conn.commit()
print(f"seeded {len(evidence_items)} evidence rows into Postgres")

# --- Weaviate ---
if weaviate_client.collections.exists("Evidence"):
    weaviate_client.collections.delete("Evidence")  # clean reseed, same spirit as TRUNCATE

weaviate_client.collections.create(
    "Evidence",
    vector_config=Configure.Vectors.self_provided(),  # bring our own MiniLM vectors, no auto-vectorizer
    properties=[
        Property(name="tenant_id", data_type=DataType.TEXT),
        Property(name="transaction_id", data_type=DataType.TEXT),
        Property(name="text", data_type=DataType.TEXT),
    ],
)

evidence_collection = weaviate_client.collections.get("Evidence")
for item in evidence_items:
    vector = embedder.encode(item.text).tolist()  # Weaviate wants a plain list, not a numpy array
    evidence_collection.data.insert(
        properties={"tenant_id": item.tenant_id, "transaction_id": item.transaction_id, "text": item.text},
        vector=vector,
    )
print(f"seeded {len(evidence_items)} evidence rows into Weaviate")