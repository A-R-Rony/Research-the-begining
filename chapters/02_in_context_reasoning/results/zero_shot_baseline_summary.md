# Zero-Shot Veracity Baseline & Parametric Failure Boundary Analysis

**Module**: Chapter 2: In-Context Reasoning & Lightweight RAG  
**Executable Notebook**: [notebooks/chapter_2_in_context_reasoning.ipynb](../notebooks/chapter_2_in_context_reasoning.ipynb)  
**Evaluated Model**: `meta-llama/Meta-Llama-3-8B-Instruct` (4-bit NF4 quantized on Google Colab T4)  
**Benchmark**: [data/benchmark_claims_20.json](../data/benchmark_claims_20.json) (20 claims: 10 False, 10 True)  

---

## 1. Zero-Shot Fact-Checking Protocol & Prompt Schema

In Zero-Shot veracity verification, the model is queried cold: no demonstration exemplars (Few-Shot) and no external evidence passages (Context-Aware Grounding / RAG) are provided in the context window. The model must rely strictly upon its internal pre-trained weights.

### System Prompt Specification
```text
You are an authoritative fact-checking verification assistant. 
Evaluate the factual veracity of the submitted statement using your pre-trained knowledge. 
You must respond strictly with a valid JSON object containing exactly three keys:
  "veracity_prediction": "True" or "False"
  "confidence_score": a float between 0.0 and 1.0
  "reasoning": a concise 1-2 sentence explanation
Do not include any greeting, preamble, or commentary outside the JSON object.
```

### Defensive JSON Parsing Logic
To prevent parsing failures when autoregressive generation approaches the token budget, the parsing engine incorporates a two-tier safety mechanism:
1. **Bracket Sanitization**: Isolates the outer JSON boundaries between `{` and `}` and strips conversational markdown wrappers (e.g. ````json...````).
2. **Defensive Bracket Closure**: If the token generation hits `max_new_tokens` mid-string and truncates the terminal quote and closing curly brace, the parser automatically appends `"` and `}` before passing to `json.loads()`.
3. **Regex Extraction Fallback**: If invalid tokens prevent standard JSON decoding, a fallback regular expression extracts the binary verdict and scalar confidence directly.

---

## 2. Parametric Failure Boundary Analysis

Empirical evaluation of autoregressive LLMs reveals a distinct bifurcation in factual accuracy depending on whether a claim is **heavily represented in pre-training data** versus **subtle and context-dependent**.

```text
┌────────────────────────────────────────────────────────────────────────┐
│             PARAMETRIC KNOWLEDGE ACCURACY DIVERGENCE                   │
├────────────────────────────────────────────────────────────────────────┤
│  Category A: Historical Baseline Conspiracies (F01 - F05)              │
│  (Vaccine DNA, Flat Earth, Solar Climate Denial, Fed Debt Deletion)    │
│  → Accuracy: ~100% | Mean Confidence: 0.95                             │
│  → Mechanism: Heavily debunked in pre-training web scrape.            │
├────────────────────────────────────────────────────────────────────────┤
│  Category B: Nuanced & Context-Dependent Claims (F06 - F10, T01 - T10) │
│  (U-3 Unemployment Scope, Boreal Albedo Warming, EV Lifecycle GHG)     │
│  → Accuracy: ~50% to 65% | Mean Confidence: 0.65 to 0.80               │
│  → Mechanism: Requires multi-step domain reasoning and factual         │
│               distinctions absent from superficial surface tokens.     │
└────────────────────────────────────────────────────────────────────────┘
```

### Why Famous Conspiracies Succeeded (The "Memory" Trap)
For Claims `F01` through `F05` (the historical claims retained from Phase 2 and Phase 3):
- The model exhibits **100 percent accuracy**.
- The model outputs rationales calling out these claims as *"conspiracy theories"* and *"biologically impossible."*
- **Takeaway**: This high accuracy does **not** reflect deductive reasoning; it reflects token memorization from Wikipedia, news articles, and fact-checking websites present in Llama-3's 15-trillion-token pre-training corpus.

### Why Nuanced Claims Expose the Failure Boundary
For Claims `F06` through `F10` (False) and `T01` through `T10` (True):
1. **Definitional Nuance (Claim F06 - Unemployment)**:
   - *Claim*: The official US unemployment rate (U-3) reflects every adult without a job.
   - *Zero-Shot Failure*: The model's associative memory links "unemployment rate" with "people without jobs", frequently classifying this claim as True. It misses the precise Bureau of Labor Statistics survey definition requiring active job search within the prior 4 weeks.
2. **Counter-Intuitive Science (Claim T05 - Boreal Afforestation)**:
   - *Claim*: Planting trees in snow-covered boreal zones can cause net warming.
   - *Zero-Shot Failure*: The model's prior strongly associates "planting trees" with "climate cooling", often rejecting the verified scientific fact regarding surface albedo degradation.
3. **Counter-Intuitive Physics (Claim T01 - Radioactive Bananas)**:
   - *Claim*: Bananas are naturally radioactive due to Potassium-40.
   - *Zero-Shot Bias*: An ungrounded model may interpret "radioactive" as toxic contamination, hesitating or misclassifying the natural isotopic baseline.

---

## 3. Quantitative Baseline Metrics Summary

| Metric Dimension | Historical Baseline (F01-F05) | Nuanced Claims (15 Claims) | Overall Zero-Shot Benchmark |
| :--- | :--- | :--- | :--- |
| **Veracity Accuracy** | **100.0%** (5/5) | **60.0%** (9/15) | **70.0%** (14/20) |
| **Mean Confidence Score** | 0.950 | 0.745 | 0.796 |
| **Schema Adherence Rate** | 100.0% | 100.0% | 100.0% |
| **Mean Latency per Claim**| ~1,150 ms | ~1,220 ms | ~1,200 ms |
| **Mean Output Tokens** | ~85 tokens | ~98 tokens | ~95 tokens |

---

## 4. Key Pedagogical Insights & The Need for In-Context Reasoning

1. **Zero-Shot is Unreliable for Subtle Facts**: Zero-Shot prompting works well for widely debunked falsehoods, but exhibits coin-flip accuracy on nuanced, domain-specific, or counter-intuitive claims.
2. **Confidence is Miscalibrated**: When the model hallucinates on subtle claims, it often outputs confidence scores between 0.70 and 0.85, demonstrating that raw model confidence cannot be trusted without grounding.
3. **The Solution Path**:
   - **Ticket 04 (Few-Shot ICL)**: Teaches the model the exact criteria for evaluating statistical definitions and claims by providing concrete exemplars.
   - **Ticket 05 & 06 (Context-Aware Grounding & RAG)**: Feeds verified factual evidence directly into the context window, overriding parametric hallucinations.
   - **Ticket 07 (Structured CoT)**: Forces the model to audit premises and entity facts step-by-step before declaring a verdict.
