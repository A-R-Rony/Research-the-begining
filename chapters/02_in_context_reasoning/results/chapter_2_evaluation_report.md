# Chapter 2 Empirical Research Report: In-Context Reasoning Paradigms, Grounding Faithfulness, and Lightweight RAG on Open-Weight Transformers

**Author**: Abdur Rony  
**Date**: October 2026  
**Target Mentorship**: Md Shoaib Ahmed (Trustworthy AI Lab, Boise State University)  
**Governing Architecture**: [ADR-0002](../../../docs/adr/0002-in-context-reasoning-and-lightweight-rag.md)  
**Specification**: [0001-open-weight-in-context-reasoning-spec.md](../../../docs/specs/0001-open-weight-in-context-reasoning-spec.md)  
**Executable Colab Notebook**: [notebooks/chapter_2_in_context_reasoning.ipynb](../notebooks/chapter_2_in_context_reasoning.ipynb)  
**Empirical Datasets**: [data/benchmark_claims_20.json](../data/benchmark_claims_20.json) | [data/knowledge_corpus.json](../data/knowledge_corpus.json) | [data/exemplars_bank.json](../data/exemplars_bank.json)  

---

## 1. Executive Summary & Research Questions

Following our replication of the social-context misinformation attack surface in Phases 0 through 3 (Ahmed & Spezzano, WSDM '26), this investigation moves beyond adversarial generation into the **reasoning mechanics of open-weight causal decoders**. Specifically, we examine how inference scaffolding alters veracity detection, factual faithfulness, and compute efficiency on `meta-llama/Meta-Llama-3-8B-Instruct`.

We designed an empirical benchmark across 20 curated claims (10 False, 10 True) spanning Health, Environment, Economy, Governance, and Technology to answer four foundational questions:

1. **RQ1 (Parametric Limits)**: At what factual boundary does cold Zero-Shot inference fail, and do 8B models exhibit over-confidence on false intuitions?
2. **RQ2 (Syntax & Induction)**: Does compiling few-shot demonstrations natively via the tokenizer chat template (`apply_chat_template`) outperform monolithic in-prompt text packing?
3. **RQ3 (Grounding & Retrieval)**: Can a lightweight in-memory RAG pipeline using PyTorch GPU matrix multiplication eliminate parametric hallucinations without external vector database bloat?
4. **RQ4 (Epistemic Reasoning vs. Compute)**: Does structured three-stage Chain-of-Thought (CoT) justify its 3.7x latency and token overhead relative to direct generation?

---

## 2. Experimental Setup & Hardware Audit

All experiments were executed within a single, reproducible Google Colab session mounted with an NVIDIA Tesla T4 GPU:

- **Base Model**: `meta-llama/Meta-Llama-3-8B-Instruct` (8.03B parameters, 32 transformer layers, 128k vocabulary).
- **Quantization**: 4-bit NormalFloat4 (NF4) with double quantization and `torch.float16` compute dtype.
- **Bi-Encoder Embedder**: `sentence-transformers/all-MiniLM-L6-v2` (22.7M parameters, 384 dimensions, GPU-resident).
- **GPU Memory Footprint**:
  - Model weights: 5.35 GB
  - Sentence embedder: 0.08 GB
  - CUDA & PyTorch workspace: 0.80 GB
  - Peak KV cache memory during inference: 0.42 GB
  - Total VRAM: **6.65 GB / 15.0 GB usable** (~8.35 GB headroom, 0 OOM errors).

---

## 3. Empirical Results: Multi-Paradigm Comparison Matrix

The table below presents the quantitative findings across all 120 recorded evaluations (20 claims evaluated across 6 conditions), exported directly from [results/benchmark_results.csv](./benchmark_results.csv) and [results/benchmark_summary_matrix.csv](./benchmark_summary_matrix.csv):

| Paradigm | Accuracy (%) | Accuracy Delta vs. Baseline | Mean Confidence | Mean Output Tokens | Mean Latency (ms) | Schema Adherence Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Zero-Shot Baseline** | **55.0%** | Baseline | 0.799 | 88.0 | 1,180.0 ms | 100.0% |
| **2. Few-Shot Native (k=3)** | **75.0%** | **+20.0%** | 0.849 | 84.0 | 1,420.0 ms | 100.0% |
| **3. Few-Shot Monolithic (k=3)** | **65.0%** | **+10.0%** | 0.798 | 112.0 | 1,560.0 ms | 90.0% (Format Drift) |
| **4. Context-Aware Grounded** | **100.0%** | **+45.0%** | 0.970 | 96.0 | 1,390.0 ms | 100.0% |
| **5. Dynamic In-Memory RAG** | **95.0%** | **+40.0%** | 0.936 | 94.0 | 1,395.0 ms | 100.0% (95% Hit@1) |
| **6. Structured Epistemic CoT** | **95.0%** | **+40.0%** | 0.947 | 382.0 | 4,410.0 ms | 100.0% |

---

## 4. Key Scientific Findings & Failure Taxonomies

### Finding 1: The Zero-Shot "Memory Trap" & The Nuance Boundary (RQ1)
Zero-Shot accuracy diverged completely based on claim popularity:
- **Historical Baseline Conspiracies (`F01`-`F05`)**: 100 percent accuracy. Llama-3 easily recognized famous debunks (microchip vaccines, solar climate denial, silver diabetes cures, Fed debt deletion, and smart meter wiretaps).
- **Nuanced Context-Dependent Claims (15 claims)**: Accuracy plunged to **40.0 percent** on subtle claims. The model exhibited severe failure modes:
  - *Definition Distortion (Claim F06)*: Believed the official US unemployment rate (U-3) counts all jobless adults, missing the active 4-week survey requirement.
  - *Associative Myth Default (Claim F08)*: Concluded that EVs generate more lifetime GHG emissions than gasoline cars, parroting common web misconceptions.
  - *Counter-Intuitive Science Rejection (Claim T05)*: Rejected the verified scientific fact that planting trees in snow-covered boreal zones causes net warming via albedo absorption.

### Finding 2: Native Chat Templates Outperform Monolithic Packing by 10% (RQ2)
Comparing the two Few-Shot implementations at `k = 3`:
- **Native Multi-Turn Turns (`apply_chat_template`)** achieved **75.0 percent accuracy** and **100.0 percent schema adherence**. By structuring exemplars between `<|start_header_id|>user` and `<|start_header_id|>assistant` tokens, the model's post-trained instruction-following circuits treated the demonstrations as authoritative prior turns.
- **Monolithic In-Prompt Packing** achieved only **65.0 percent accuracy** and dropped to **90.0 percent schema adherence**. Packing all demonstrations into a single user turn induced **pattern over-extension**: the model echoed "Example 4:" and emitted conversational preambles that broke JSON parsing.

### Finding 3: Context-Aware Grounding Eliminates Hallucination with 100% Flip Rate (RQ3)
Injecting authoritative background passages from `data/knowledge_corpus.json` into the context envelope produced an unprecedented leap:
- Accuracy increased from **55.0 percent to 100.0 percent** (+45.0 percentage points).
- **100 Percent Error Flip Rate**: Every single claim that failed under Zero-Shot flipped to the correct verdict.
- **Faithfulness**: When provided with the BLS and ICCT reference texts, the model cited the exact factual criteria (the 4-week active search rule, and the 1-to-2 year EV break-even point), successfully overriding its pre-trained associative biases.

### Finding 4: Pure PyTorch Matrix Search Enables Sub-Millisecond Dynamic RAG (RQ3)
In Stage B, rather than pre-pairing documents, the system performed dynamic semantic retrieval using `all-MiniLM-L6-v2`:
- **Retrieval Hit@1 Rate**: **95.0 percent** (19 of 20 claims retrieved their exact matching document; the 1 partial mismatch retrieved a related lifecycle document).
- **Retrieval Latency**: GPU matrix multiplication (`torch.mm`) over the normalized `[20, 384]` document matrix executed in **0.18 milliseconds**.
- **End-to-End RAG Accuracy**: **95.0 percent**, operating within 5 milliseconds of direct inference while eliminating external vector database dependencies (FAISS, ChromaDB).

### Finding 5: Structured Epistemic CoT Resolves Logic at a 3.7x Compute Cost (RQ4)
Our three-stage epistemic decomposition (`source_and_entity_check` → `evidence_triangulation` → `fallacy_or_distortion_detection`) elevated ungrounded accuracy from 55.0 percent to **95.0 percent**:
- Forcing the model to audit entities and identify distortions *before* emitting the verdict token enabled it to self-correct reasoning traps.
- **The Compute Cost**: CoT generated an average of **382.0 output tokens** per claim compared to 88.0 for Zero-Shot, increasing wall-clock generation latency from 1,180 ms to **4,410 ms (3.7x slower)**.

---

## 5. Systems Efficiency & The Pareto Frontier

When architecting AI verification systems in production or laboratory environments, engineers must navigate the trade-off between factual accuracy and inference latency:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   ACCURACY VS. COMPUTE PARETO FRONTIER                 │
├────────────────────────────────────────────────────────────────────────┤
│  100% ┤                                 [Context Grounded]             │
│       │                                      (1,390ms)                 │
│   95% ┤                                 [Dynamic RAG]   [Epistemic CoT]│
│       │                                    (1,395ms)        (4,410ms)  │
│   75% ┤              [Few-Shot Native]                                 │
│       │                  (1,420ms)                                     │
│   65% ┤              [Few-Shot Mono]                                   │
│       │                  (1,560ms)                                     │
│   55% ┤ [Zero-Shot]                                                    │
│       │   (1,180ms)                                                    │
│   50% └─────┬──────────────┬──────────────────┬──────────────┬──────── │
│           1,000ms        1,500ms            2,000ms        4,500ms     │
│                             WALL-CLOCK INFERENCE LATENCY               │
└────────────────────────────────────────────────────────────────────────┘
```

### The Pareto-Optimal Architecture
The empirical sweet spot is **Dynamic In-Memory RAG**:
- It achieves **95.0 percent accuracy** (matching CoT).
- It executes in **1,395 ms** (only 18% slower than cold Zero-Shot, and **68% faster than CoT**).
- It generates only 94 tokens per query, preserving GPU memory bandwidth and minimizing operational compute costs.

---

## 6. Alignment with Md Shoaib Ahmed's Trustworthy AI Lab

This benchmark provides immediate, actionable extensions to the misinformation attack surface established in ACM WSDM '26:

1. **Defending Detectors Against Adversarial Astroturfing**: In Phase 3, we proved that social-context detectors can be influenced by polite, synthetic adversarial comments. The findings of Chapter 2 demonstrate that coupling detectors with **in-memory RAG grounding** inoculates the model: when an authoritative evidence passage is present, the model prioritizes factual ground truth over deceptive social consensus.
2. **Epistemic Scaffolding for Social Deception**: When external evidence is unavailable, employing **three-stage epistemic CoT** forces the detector to audit whether community comments exhibit anecdotal fallacies or institutional cynicism, neutralizing the polite astroturfing techniques identified in our linguistic taxonomy.
3. **Open-Weight Autonomy**: The entire pipeline (bi-encoder retrieval + 4-bit Llama-3-8B generation) operates 100% locally on free Google Colab T4 hardware, providing zero-cost scalability for benchmarking thousands of claims.

---

## 7. Deliverable Artifacts & Navigation

- **Chapter Syllabus**: [chapters/02_in_context_reasoning/README.md](../README.md)
- **Executable Lab Notebook**: [notebooks/chapter_2_in_context_reasoning.ipynb](../notebooks/chapter_2_in_context_reasoning.ipynb)
- **Raw Evaluation Data (120 rows)**: [results/benchmark_results.csv](./benchmark_results.csv)
- **Summary Matrix**: [results/benchmark_summary_matrix.csv](./benchmark_summary_matrix.csv)
- **Theory Modules**:
  - [Topic 2.1: Open-Weight Prompt Serialization](../notes/topic_2.1_open_weight_prompt_mechanics.md)
  - [Topic 2.2: Zero-Shot & Few-Shot ICL](../notes/topic_2.2_zero_shot_and_few_shot_icl.md)
  - [Topic 2.3: Context-Aware Grounding & RAG](../notes/topic_2.3_context_grounding_and_rag.md)
  - [Topic 2.4: Structured Chain-of-Thought](../notes/topic_2.4_chain_of_thought_reasoning.md)
