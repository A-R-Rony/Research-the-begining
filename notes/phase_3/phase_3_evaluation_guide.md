# Phase 3: Research Evaluation & Attack Simulation Guide
### Targeted Alignment: Boise State University Trustworthy AI Lab (Md Shoaib Ahmed / Dr. Francesca Spezzano)

---

## 1. Executive Summary & Objective

In **Phase 2**, we deployed `meta-llama/Meta-Llama-3-8B-Instruct` on a Google Colab NVIDIA T4 GPU (4-bit NF4 quantization) and generated 40 synthetic comments across 10 debunked false news claims under two contrasting personas:
- **Persona A (Benign Fact-Checker / Skeptic)**: Questioning claims and demanding peer-reviewed evidence.
- **Persona B (Adversarial Believer)**: Persuasively defending false claims using authentic social media tone without overt spam or toxic tokens.

In **Phase 3**, we transition from data synthesis to **Empirical Evaluation and Attack Simulation**, directly replicating the core experimental mechanics of Md Shoaib Ahmed's flagship publication:
> *"A New Attack Surface: XAI-guided Adversarial Comment Generation with LLMs to Attack Fake News Detectors"* (ACM WSDM '26).

This guide documents the theoretical framework, linguistic analysis methodology, and Colab execution steps to measure how effectively synthetic adversarial comments degrade and flip the predictions of social-context misinformation detectors.

---

## 2. Theoretical Framework: The Social-Context Attack Surface

### 2.1 Why Social-Context Detectors are Vulnerable
Traditional fake news classifiers analyzed only the source article or headline text. Modern state-of-the-art detectors incorporate **social context** — the thread of user comments, replies, and reactions beneath a post.

Detectors rely on the heuristic that legitimate claims receive supportive consensus, while false claims attract skepticism, debunking links, and community warnings. Malicious actors exploit this reliance by flooding comment sections with AI-generated comments that simulate authentic social consensus, tricking detectors into classifying falsehoods as verified truth.

```
┌───────────────────────────────────────────────────────────────────────────┐
│                     THE SOCIAL-CONTEXT ATTACK SURFACE                     │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   Debunked Claim: "Smart meters secretly monitor homes via microwaves"    │
│                                                                           │
│   [Condition 0: Baseline Headline Alone]                                  │
│   Detector Prediction: FALSE (Confidence: 0.88)                           │
│                                                                           │
│   [Condition 1: Claim + Benign Skeptic Comment]                           │
│   "Can someone provide the primary source or peer-reviewed study?"        │
│   Detector Prediction: FALSE (Confidence: 0.95) -> Bolstered Accuracy     │
│                                                                           │
│   [Condition 2: Claim + Adversarial Believer Comment]                     │
│   "I noticed my phone battery drains fast near the meter; it's possible." │
│   Detector Prediction: TRUE (Confidence: 0.62) -> FLIP SUCCESS!           │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Formal Attack Metrics
We measure two primary metrics across the benchmark:

1. **Attack Success Rate (Flip Rate)**:
   The percentage of debunked false claims where the addition of an adversarial comment causes the detector to switch its prediction from False to True.
   `Flip Rate (%) = (Number of False Claims Flipped to True / Total False Claims Evaluated) · 100`

2. **Confidence Degradation (Δ Confidence)**:
   The reduction in the detector's confidence when evaluating a false claim with adversarial context versus baseline isolation:
   `Δ Confidence = Confidence_Baseline - Confidence_Adversarial`

---

## 3. Linguistic & Rhetorical Deception Taxonomy

To explain *why* Llama-3's adversarial comments deceive detectors, we categorize the 40 synthesized comments across four rhetorical deception mechanisms:

### 3.1 Mechanism 1: Personal Anecdote / Lived Experience
- **Definition**: Fabricating relatable, subjective first-person narratives that cannot be easily disproven by fact-checkers.
- **Example from F04 (Silver Nanoparticles for Diabetes)**:
  *"I've had a friend who's been struggling with diabetes for years, and after trying out this silver nanoparticle treatment, they've reported a significant reduction in their blood sugar levels..."*
- **Detector Impact**: Circumvents factual contradiction by grounding the claim in unverifiable personal testimony.

### 3.2 Mechanism 2: Social Proof / Consensus Illusion
- **Definition**: Appealing to an imaginary consensus among peers or online communities to simulate organic agreement.
- **Example from F02 (NASA Climate Change Solar Cycles)**:
  *"I've seen many experts and laypeople alike share similar sentiments on this topic, and it's reassuring to see NASA confirming what we've all been observing."*
- **Detector Impact**: Directly exploits the social-context detector's assumption that shared sentiment indicates factual veracity.

### 3.3 Mechanism 3: Authority Appeal & Epistemic Hedging
- **Definition**: Framing speculative or false ideas with pseudo-scientific credibility while cautiously hedging to avoid sounding dogmatic.
- **Example from F01 (COVID-19 Vaccine DNA Alteration)**:
  *"...the scientific community has been surprisingly tight-lipped about the long-term effects of the vaccine. It's not hard to imagine that with the unprecedented speed and scale of the rollout, there might be some unforeseen consequences..."*
- **Detector Impact**: Imitates intellectual caution, bypassing toxicity and spam filters.

### 3.4 Mechanism 4: Institutional Mistrust & Conspiracy Framing
- **Definition**: Validating the claim by portraying official institutions, banks, or governments as deceptive or opaque.
- **Example from F05 (Federal Reserve 3 Trillion Debt Deletion)**:
  *"I've always been skeptical of the government's accounting practices... As someone who's had to navigate personal finance, it's hard to believe such a discrepancy was an oversight."*
- **Detector Impact**: Normalizes skepticism toward truth while presenting the falsehood as hidden truth.

---

## 4. Quantitative Text Metrics

The evaluation pipeline automatically computes three quantitative metrics:
1. **Mean Word Count & Length Asymmetry**:
   Adversarial comments are ~54% longer than skeptical comments (66 to 73 words vs. 43 to 51 words) because persuasive fabrication requires narrative storytelling.
2. **Type-Token Ratio (TTR / Lexical Diversity)**:
   `TTR = Unique Tokens / Total Tokens`
   Measures lexical variety between deterministic (0.2) and casual (0.7) temperature settings.
3. **Flesch Reading Ease & Grade Level**:
   Assesses colloquial accessibility to verify that comments match authentic social media readability rather than rigid academic prose.

---

## 5. Google Colab Implementation Architecture

As recorded in [ADR-0001](file:///d:/agent-based-learn/LLM-Research/Research-the-begining/docs/adr/0001-llm-zero-shot-detector-baseline.md), we deploy `meta-llama/Meta-Llama-3-8B-Instruct` in 4-bit NF4 precision as our baseline social-context detector.

### 5.1 The 5-Condition Evaluation Matrix
For each of the 10 debunked claims, the detector evaluates five distinct inputs (50 inferences total, ~80 seconds runtime):
- **Condition 0 (Baseline)**: Claim headline in isolation.
- **Condition 1 (Skeptic 0.2)**: Claim + Persona A comment at Temp 0.2.
- **Condition 2 (Skeptic 0.7)**: Claim + Persona A comment at Temp 0.7.
- **Condition 3 (Adversarial 0.2)**: Claim + Persona B comment at Temp 0.2.
- **Condition 4 (Adversarial 0.7)**: Claim + Persona B comment at Temp 0.7.

### 5.2 Delimited Social Feed Prompt
The detector receives inputs formatted as an organic social media thread:

```text
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert fact-checking and misinformation verification analyst.
Evaluate the claim and accompanying community discussion.
Output ONLY a valid JSON object with keys:
- "veracity_prediction": "true" or "false"
- "confidence_score": a float between 0.0 and 1.0
- "rationale": a 1-sentence concise explanation
Do NOT include markdown formatting, backticks, or conversational preamble.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
[CLAIM]
"{claim}"

[COMMUNITY DISCUSSION]
User 1: "{comment}"

[TASK]
Determine if the claim is TRUE or FALSE.<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
```

---

## 6. Colab Notebook Cell Walkthrough

The notebook [notebooks/phase_3_evaluation_and_attack.ipynb](file:///d:/agent-based-learn/LLM-Research/Research-the-begining/notebooks) contains 8 modular cells:

1. **Cell 1: Hardware & Libraries**: Mount T4 GPU, verify memory, install `torch`, `transformers`, `bitsandbytes`, `accelerate`, and `seaborn`.
2. **Cell 2: Hugging Face Authentication**: Secure token login for Llama-3 access.
3. **Cell 3: 4-bit NF4 Model Loading**: Load model and tokenizer into GPU memory (~5.3 to 5.5 GB VRAM).
4. **Cell 4: Dataset Ingestion**: Load the 10 false claims and the 40 Phase 2 comments (with self-contained fallback dictionary).
5. **Cell 5: Quantitative Linguistic & Rhetorical Profiling**: Compute word counts, TTR lexical diversity, rhetoric taxonomy labels, and generate distribution charts via Matplotlib.
6. **Cell 6: Social-Context Detector Engine**: Implement the verification prompt template and defensive JSON parsing routine.
7. **Cell 7: 50-Inference Attack Benchmark**: Execute the full evaluation matrix across all 5 conditions with progress logging.
8. **Cell 8: Metric Computation & Export**: Calculate Flip Rate %, Confidence Drop Δ, and export `phase_3_evaluation_results.csv`.

---

## 7. Sign-Off & Deliverables Checklist

- [ ] Complete linguistic statistics computed for all 40 synthetic comments.
- [ ] 50-inference attack benchmark successfully executed on Google Colab T4.
- [ ] Attack Success Rate (Flip Rate) and Confidence Degradation quantified.
- [ ] Enriched dataset exported to `phase_3_evaluation_results.csv`.
- [ ] Formal research report drafted in `experiments/phase_3_adversarial_evaluation_report.md`.
