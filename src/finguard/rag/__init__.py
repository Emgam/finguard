"""FinGuard knowledge base (Phase 3): registry, ingestion, hybrid retrieval, abstention, evaluation.

The knowledge base answers three questions about an alert: what the pattern means, what to
check next, and what the institution's obligations are if fraud is confirmed. It never
"verifies a transaction against regulation". Every passage it returns carries its document,
publisher, date, jurisdiction, section and page.

    python -m finguard.rag build                  # parse, chunk and index every registered document
    python -m finguard.rag search "query" [--tier 1] [--jurisdiction EU]
    python -m finguard.rag evaluate               # Hit@5 / MRR / abstention on the frozen question set
"""
