# 02: Curate 20-Claim Benchmark Suite and Knowledge Corpus

**What to build:** A balanced, transparent benchmark dataset and accompanying evidence corpus stored in clean JSON format, containing 5 historical baseline claims from Phase 2/3 and 15 subtle context-dependent claims requiring factual reasoning, along with high-trust evidence passages and standardized few-shot exemplars.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] Assemble 20 balanced veracity claims (10 False, 10 True) spanning Health, Science, Economy, Governance, and Technology in `data/benchmark_claims_20.json`.
- [x] Retain 5 historical claims from Phase 2 to provide direct backward regression comparison against Phase 3 baselines.
- [x] Curate verified, neutral background evidence passages for each claim in `data/knowledge_corpus.json` for grounding and RAG testing.
- [x] Assemble structured multi-turn few-shot exemplars in `data/exemplars_bank.json` covering True and False claim demonstrations.

