# 05: Context-Aware Evidence Grounding and Faithfulness Verification

**What to build:** An evidence-grounded inference harness and pedagogical guide demonstrating how injecting external factual passages into the context window forces the model to prioritize supplied evidence over internal parametric memory, measuring evidence faithfulness and resistance to hallucination.

**Blocked by:** 02 (Curate 20-Claim Benchmark Suite and Knowledge Corpus), 03 (Zero-Shot Veracity Baseline and Failure Boundary Analysis)

**Status:** done

- [x] Write conceptual guide on Context-Aware Grounding, defining parametric belief collision and epistemic faithfulness.
- [x] Implement grounded prompt envelope injecting claim-specific evidence passages from `data/knowledge_corpus.json` into the model's context window.
- [x] Measure veracity flip rates and confidence shifts when evidence directly contradicts initial zero-shot beliefs.
- [x] Compute a quantitative evidence faithfulness metric verifying whether model rationales cite facts strictly present in the injected passage.

