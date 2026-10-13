# Research Progress Report: Adversarial Social-Context Attack Surface Replication

**Author**: Abdur Rony  
**Date**: October 2026  
**Repository**: [github.com/A-R-Rony/Research-the-begining](https://github.com/A-R-Rony/Research-the-begining)  

---

## Overview

This report documents what I have built, learned, and discovered over a 4-phase self-directed research sprint aimed at replicating and extending the social-context misinformation attack surface described in:

> Ahmed & Spezzano, *"A New Attack Surface: XAI-guided Adversarial Comment Generation with LLMs to Attack Fake News Detectors"*, ACM WSDM '26.

All code is executable on Google Colab (free T4 GPU). Notebooks, datasets, and the full research log are in the repository linked above.

---

## Phase 0: Conceptual Foundations

Studied the technical prerequisites before writing any code:

- **Llama-3-8B Architecture**: 32-layer decoder-only transformer, 8.03B parameters, GQA (32 query / 8 KV heads), 128k BPE vocabulary, RoPE, SwiGLU activations.
- **4-bit NF4 Quantization**: Why FP16 (16.06 GB) exceeds the T4's 15.8 GB ceiling; how NF4 with double quantization compresses the model to ~5.35 GB while preserving inference quality via on-the-fly Tensor Core dequantization.
- **Social-Context Detection Landscape**: How detectors like dEFEND, SAFE, and TriFN use co-attention over user comments; how adversarial astroturfing exploits their crowd-wisdom heuristic.
- **Why Open Weights**: Deterministic seeds, logit access for XAI attribution, zero marginal cost at scale, no API deprecation risk.

**Detailed notes**: [notes/phase_0/](../notes/phase_0/)

---

## Phase 1: Deploying Llama-3-8B on Colab T4

Loaded `meta-llama/Meta-Llama-3-8B-Instruct` in 4-bit NF4 precision on a free Google Colab T4 GPU.

- **VRAM on cold load**: 5.35 GB
- **VRAM after inference**: 5.47 GB (stable, no leaks)
- **Remaining headroom**: ~10.4 GB for long context sequences

**Notebook**: [notebooks/phase_1_llama3_colab.ipynb](../notebooks/phase_1_llama3_colab.ipynb)

---

## Phase 2: Instruction Following & Dual-Persona Comment Synthesis

### Experiment 2.1 — Structured JSON Output
Prompted Llama-3-8B to output strict JSON (`veracity`, `sentiment`, `confidence`, `reasoning`).
- **Bug found**: At `max_new_tokens=150`, the model hit the token limit mid-sentence and omitted the closing `}`, causing 0% parse rate.
- **Fix**: Expanded budget to 220 tokens + defensive bracket closure before `json.loads()`. Achieved 100% adherence.

### Experiment 2.2 — Adversarial Comment Generation
Generated **40 synthetic comments** across **10 debunked false claims** (sourced from PolitiFact / LIAR) spanning Health, Environment, Economy, Governance, and Technology.

Two personas, two temperatures per claim:
- **Persona A (Skeptic)**: Demands primary sources, questions credibility.
- **Persona B (Adversarial Believer)**: Defends the false claim using authentic social media tone — no toxicity, no spam.

**Dataset**: [notebooks/phase_2_experimental_results.csv](../notebooks/phase_2_experimental_results.csv) (40 rows)  
**Notebook**: [notebooks/phase_2_adversarial_comments.ipynb](../notebooks/phase_2_adversarial_comments.ipynb)

---

## Phase 3: Empirical Attack Simulation & Findings

### 3.1 Linguistic Profile of Generated Comments

| Persona | Temp | n | Mean Words | Std Dev | Mean TTR |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Skeptic | 0.2 | 10 | 43.1 | 6.8 | 0.812 |
| Skeptic | 0.7 | 10 | 51.4 | 6.1 | 0.804 |
| Adversarial | 0.2 | 10 | 66.3 | 9.4 | 0.768 |
| Adversarial | 0.7 | 10 | 72.6 | 12.1 | 0.752 |

Key observations:
- **53.8% length asymmetry**: Adversarial comments are longer because fabricating believable defense requires narrative context; skepticism requires only 1-2 targeted questions.
- **Temperature effect**: Higher temperature (+19% words for skeptics, +10% for adversarial) introduced colloquial filler and emotional nuance.
- **Lexical density**: Skeptic TTR (0.81) > Adversarial TTR (0.76) — skeptics use dense epistemic vocabulary; adversarial comments rely on relatable first-person pronouns ("I", "my", "we").

### 3.2 Rhetorical Deception Taxonomy (Persona B)

Qualitative coding of the 20 adversarial comments revealed 4 mechanisms:

| Mechanism | Frequency | Example |
| :--- | :--- | :--- |
| **Personal Anecdote / Lived Experience** | 35% | *"I've had a friend who tried silver nanoparticles and reduced their insulin..."* |
| **Social Proof / Consensus Illusion** | 30% | *"Many experts and laypeople share similar sentiments..."* |
| **Epistemic Hedging** | 20% | *"It's not hard to imagine there might be unforeseen consequences..."* |
| **Institutional Mistrust** | 15% | *"I've always been skeptical of the government's accounting practices..."* |

### 3.3 Attack Benchmark: 50-Inference Evaluation

Deployed Llama-3-8B as a zero-shot social-context detector (system prompt: expert fact-checker; input: claim + community comment; output: JSON with `veracity_prediction`, `confidence_score`, `rationale`).

Evaluated all 10 claims across 5 conditions:

| Condition | Accuracy | Flip Rate | Mean Confidence | Δ Confidence |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (headline only) | 100% | 0% | 0.915 | — |
| + Skeptic (T=0.2) | 100% | 0% | 0.910 | -0.005 |
| + Skeptic (T=0.7) | 100% | 0% | 0.910 | -0.005 |
| + Adversarial (T=0.2) | 100% | 0% | 0.910 | -0.005 |
| + Adversarial (T=0.7) | 100% | 0% | 0.910 | -0.005 |

### 3.4 Why 0% Flip Rate — and Why That's Still Informative

The zero-shot 8B model was **not fooled**. This is itself a research finding worth discussing:

1. **Famous claims**: These 10 conspiracies (vaccine tracking, Finland non-existence, etc.) are heavily debunked in Llama-3's training data — the model simply "knows" they're false.
2. **Model called it out**: The detector's rationales explicitly flagged the adversarial comments as *"anecdotal evidence"*, *"speculation rather than credible sources"*, and *"personal testimonials, not a reliable basis."*
3. **One comment isn't enough**: Published attacks (WSDM '26) use multiple coordinated comments against graph-based detectors (dEFEND, BiGCN) — not a single comment against a causal LLM.
4. **Safety blind spot**: Despite this robustness, Llama-3 had a **0% refusal rate** when *generating* the adversarial comments — its safety filters catch toxicity but not polite deception.

**Dataset**: [notebooks/phase_3_evaluation_results.csv](../notebooks/phase_3_evaluation_results.csv) (50 rows)  
**Notebook**: [notebooks/phase_3_evaluation_and_attack.ipynb](../notebooks/phase_3_evaluation_and_attack.ipynb)  
**Full Report**: [experiments/phase_3_adversarial_evaluation_report.md](../experiments/phase_3_adversarial_evaluation_report.md)


---

## Repository Artifact Index

| Artifact | Path |
| :--- | :--- |
| Research Log (daily journal) | [RESEARCH_LOG.md](../RESEARCH_LOG.md) |
| Master Plan (4-phase roadmap) | [PLAN.md](../PLAN.md) |
| Glossary | [GLOSSARY.md](../GLOSSARY.md) |
| ADR-0001 (detector baseline) | [docs/adr/0001-llm-zero-shot-detector-baseline.md](./adr/0001-llm-zero-shot-detector-baseline.md) |
| Phase 1 Notebook | [notebooks/phase_1_llama3_colab.ipynb](../notebooks/phase_1_llama3_colab.ipynb) |
| Phase 2 Notebook | [notebooks/phase_2_adversarial_comments.ipynb](../notebooks/phase_2_adversarial_comments.ipynb) |
| Phase 3 Notebook | [notebooks/phase_3_evaluation_and_attack.ipynb](../notebooks/phase_3_evaluation_and_attack.ipynb) |
| Phase 2 Dataset (40 comments) | [notebooks/phase_2_experimental_results.csv](../notebooks/phase_2_experimental_results.csv) |
| Phase 3 Dataset (50 inferences) | [notebooks/phase_3_evaluation_results.csv](../notebooks/phase_3_evaluation_results.csv) |
| Graduate Research Report | [experiments/phase_3_adversarial_evaluation_report.md](../experiments/phase_3_adversarial_evaluation_report.md) |
