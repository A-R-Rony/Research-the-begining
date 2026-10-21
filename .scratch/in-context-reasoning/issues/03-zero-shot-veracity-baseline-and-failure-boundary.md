# 03: Zero-Shot Veracity Baseline and Failure Boundary Analysis

**What to build:** An end-to-end zero-shot veracity detection pipeline running on `meta-llama/Meta-Llama-3-8B-Instruct` in 4-bit precision, establishing a clean baseline of what the model knows strictly via its internal parametric weights and identifying its failure boundaries on ambiguous claims.

**Blocked by:** 01 (Open-Weight Prompt Serialization and KV Cache Memory Guide), 02 (Curate 20-Claim Benchmark Suite and Knowledge Corpus)

**Status:** done

- [x] Construct a zero-shot system prompt and structured JSON response contract (`veracity_prediction`, `confidence_score`, `rationale`).
- [x] Implement defensive parsing to guarantee 100 percent JSON extraction without unclosed token truncation.
- [x] Evaluate all 20 claims from `data/benchmark_claims_20.json` under cold parametric inference.
- [x] Quantify the accuracy divergence between memorized historical claims versus subtle context-dependent claims.

