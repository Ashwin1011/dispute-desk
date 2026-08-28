-- DisputeDesk database schema.
-- Run once against a fresh Postgres instance with the pgvector extension available
-- (e.g. `psql -f schema.sql`, or via the pgvector/pgvector Docker image in CI).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS evidence (
    tenant_id text,
    transaction_id text,
    text text,
    embedding vector(384),
    text_search tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS evidence_text_search_idx ON evidence USING GIN (text_search);

CREATE TABLE IF NOT EXISTS audit_log (
    id serial PRIMARY KEY,
    thread_id text,
    transaction_id text,
    node_name text,
    decision jsonb,
    created_at timestamptz DEFAULT now()
);