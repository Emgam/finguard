"""FinGuard agents (Phase 4): an LLM orchestrator, an investigation agent and a report agent
working inside a contract they cannot break (Project Report v3.0, section 8).

* ``tools``     the six tools with JSON schemas (fraud_score, explain_transaction, entity_history,
                indicator_map, policy_search, case_store) and the evidence they produce;
* ``llm``       LLM clients: Anthropic, any OpenAI-compatible endpoint (OpenAI, Groq, Ollama ...),
                and a deterministic scripted model for tests;
* ``agents``    the three LLM agents: orchestrator (plan), investigation, report;
* ``policy``    the evidence-policy validator: code, not prompts;
* ``report``    facts by template, the fallback report, Markdown rendering;
* ``graph``     the state graph (LangGraph when installed, a built-in engine otherwise);
* ``evaluate``  the 20 scripted cases and the agent metrics.

Run ``python -m finguard.agents --help``.
"""
