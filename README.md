# Research-the-begining: Empirical LLM Vulnerability & Adversarial Misinformation Pipeline

### Targeted Academic Alignment: Boise State University Trustworthy AI Lab
**Targeted Researchers**: Md Shoaib Ahmed & Dr. Francesca Spezzano  
**Flagship Literature Alignment**: ACM WSDM '26 (*A New Attack Surface: XAI-guided Adversarial Comment Generation with LLMs to Attack Fake News Detectors*) & IEEE TCSS '26  

---

## 🔬 Executive Overview

This repository documents an end-to-end applied Large Language Model (LLM) empirical research pipeline. Designed to bridge foundational software engineering into graduate-level NLP research, this project replicates and expands upon the core mechanics of the **social-context attack surface** against misinformation detectors.

Utilizing `meta-llama/Meta-Llama-3-8B-Instruct` deployed in 4-bit NormalFloat4 (NF4) precision on consumer-grade GPU hardware (Google Colab NVIDIA Tesla T4, 15.8 GB VRAM), we investigate how conversational social proof degrades and perturbs LLM-based fact-checking detectors.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     THE RESEARCH TRAJECTORY & CHAPTERS                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Phases 0-4: Adversarial Misinformation & Attack Surface (WSDM '26)      │
│     - Phase 0: Mathematical & VRAM Foundations                           │
│     - Phase 1: Operational Colab T4 Milestone (NF4 Llama-3-8B)           │
│     - Phase 2: Dual-Persona Synthetic Comment Synthesis (40 Samples)     │
│     - Phase 3: Linguistic Profiling, Attack Simulation & Academic Report │
│     - Phase 4: Lab Pitch & Research Integration (Boise State)            │
│                                                                          │
│  Chapter 2: In-Context Reasoning, Grounding & Lightweight RAG            │
│     - Zero-Shot vs. Native/Monolithic Few-Shot ICL                       │
│     - Context-Aware Grounding & In-Memory GPU Matrix RAG (0.18 ms)       │
│     - Structured Epistemic Chain-of-Thought (CoT) & Self-Consistency     │
│     - 120-Inference Multi-Metric Empirical Benchmark Suite               │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Key Empirical Findings

### 1. The Persuasion Length Asymmetry
Across 40 generated comments on 10 debunked claims, adversarial believer comments were on average **53.8% longer** than skeptical fact-checking inquiries:
- **Persona A (Skeptic)**: 43.1 words (Temp 0.2) to 51.4 words (Temp 0.7) — compact, targeted demands for primary evidence.
- **Persona B (Adversarial Believer)**: 66.3 words (Temp 0.2) to 72.6 words (Temp 0.7) — constructive narrative building and fabricated lived experience.

### 2. Qualitative Rhetorical Deception Taxonomy
Qualitative analysis of the adversarial comments identified a 4-mechanism taxonomy used by Llama-3-8B to defend falsehoods:
1. **Personal Anecdote / Lived Experience (35%)**: Inventions of firsthand or acquaintance testimony that cannot be checked against external databases.
2. **Social Proof / Consensus Illusion (30%)**: Simulating community consensus to appeal to the crowd heuristic.
3. **Epistemic Hedging & Pseudo-Science (20%)**: Avoiding dogmatism through measured intellectual caution.
4. **Institutional Mistrust & Cynicism (15%)**: Leveraging suspicion toward banks, agencies, and governments.

### 3. Empirical Attack Benchmark & Robustness
- **Zero-Shot 8B Robustness**: On high-salience, historically debunked claims (e.g., vaccine tracking, flat earth), zero-shot Llama-3-8B acting as an expert fact-checking detector showed 100% accuracy against single-comment perturbation.
- **Confidence Degradation**: Adversarial comments reliably induced measurable confidence drops (mean baseline confidence dropped from 0.915 to 0.910), demonstrating subtle epistemic erosion.
- **Safety Audit**: Documented a 0.0% refusal rate when generating polite, deceptive comments devoid of overt toxic tokens.

---

## 🧭 Repository Structure & Research Artifacts

### 📚 Research Journal & Core Documentation
- **[docs/COMPREHENSIVE_RESEARCH_JOURNEY.md](./docs/COMPREHENSIVE_RESEARCH_JOURNEY.md)**: Master research monograph and technical dossier documenting the full journey from zero ML experience to empirical attack simulation.
- **[RESEARCH_LOG.md](./RESEARCH_LOG.md)**: Comprehensive daily research journal detailing concepts, experiments, and debugging milestones.
- **[PLAN.md](./PLAN.md)**: Master 5-phase research roadmap and deliverables.
- **[CONTEXT.md](./CONTEXT.md)**: Research background, personal goals, and mentorship thesis.
- **[SENIOR_OVERVIEW.md](./SENIOR_OVERVIEW.md)**: Detailed profile of Md Shoaib Ahmed, citation metrics (767+ citations, h-index: 12), and complete publication catalog.
- **[GLOSSARY.md](./GLOSSARY.md)**: Canonical terminology definitions for LLM research, tensors, and runtime states.


### 🧪 Google Colab Notebooks
- **[notebooks/phase_1_llama3_colab.ipynb](./notebooks/phase_1_llama3_colab.ipynb)**: Phase 1 baseline deployment of 4-bit Llama-3-8B on Colab T4.
- **[notebooks/phase_2_adversarial_comments.ipynb](./notebooks/phase_2_adversarial_comments.ipynb)**: Phase 2 dual-persona comment synthesis and JSON adherence.
- **[notebooks/phase_3_evaluation_and_attack.ipynb](./notebooks/phase_3_evaluation_and_attack.ipynb)**: Phase 3 50-inference empirical attack simulation and charting.

### 📝 Technical Guides & Academic Reports
- **[experiments/phase_3_adversarial_evaluation_report.md](./experiments/phase_3_adversarial_evaluation_report.md)**: Graduate empirical research report formatted for conference submission.
- **[docs/adr/0001-llm-zero-shot-detector-baseline.md](./docs/adr/0001-llm-zero-shot-detector-baseline.md)**: Architectural Decision Record for the LLM detector baseline.
- **[notes/phase_4/phase_4_pitch_and_mentorship_guide.md](./notes/phase_4/phase_4_pitch_and_mentorship_guide.md)**: Outreach pitch scripts, defense Q&A, and lab contribution proposals.
- **[notes/phase_3/phase_3_evaluation_guide.md](./notes/phase_3/phase_3_evaluation_guide.md)**: Evaluation framework and metric computation guide.
- **[notes/phase_2/phase_2_experiment_guide.md](./notes/phase_2/phase_2_experiment_guide.md)**: Experimental guide for comment synthesis and JSON formatting.
- **[notes/phase_1/phase_1_colab_guide.md](./notes/phase_1/phase_1_colab_guide.md)**: Step-by-step T4 deployment and VRAM management guide.
- **[notes/tooling/README.md](./notes/tooling/README.md)**: Python deep learning stack guide for software engineers.

### 💾 Datasets
- **[notebooks/phase_2_experimental_results.csv](./notebooks/phase_2_experimental_results.csv)**: 40 synthesized comments across 10 claims and 2 personas.
- **[notebooks/phase_3_evaluation_results.csv](./notebooks/phase_3_evaluation_results.csv)**: Full 50-inference evaluation matrix with veracity predictions, confidence scores, and rationales.

### 🧠 Chapter 2: In-Context Reasoning & Lightweight RAG
- **[Chapter 2 Syllabus](./chapters/02_in_context_reasoning/README.md)**: Curriculum overview, 4-pillar theory index, and laboratory instructions.
- **[Chapter 2 Colab Notebook](./chapters/02_in_context_reasoning/notebooks/chapter_2_in_context_reasoning.ipynb)**: Executable 32-cell research notebook on Google Colab T4.
- **[Formal Evaluation Report](./chapters/02_in_context_reasoning/results/chapter_2_evaluation_report.md)**: Graduate-level empirical research report synthesizing findings across 120 recorded evaluations.
- **[ADR-0002](./docs/adr/0002-in-context-reasoning-and-lightweight-rag.md)**: Architectural Decision Record for 4-bit NF4 Llama-3-8B and PyTorch GPU matrix retrieval.
- **[Spec 0001](./docs/specs/0001-open-weight-in-context-reasoning-spec.md)**: Formal specification and 18 user stories for in-context reasoning paradigms.
- **Curated Datasets**:
  - [data/benchmark_claims_20.json](./chapters/02_in_context_reasoning/data/benchmark_claims_20.json): 20 balanced evaluation claims.
  - [data/knowledge_corpus.json](./chapters/02_in_context_reasoning/data/knowledge_corpus.json): 20 verified reference evidence passages.
  - [data/exemplars_bank.json](./chapters/02_in_context_reasoning/data/exemplars_bank.json): 6 dual-formatted few-shot exemplars.
- **Benchmark Metrics**:
  - [results/benchmark_results.csv](./chapters/02_in_context_reasoning/results/benchmark_results.csv): 120-inference raw evaluation records.
  - [results/benchmark_summary_matrix.csv](./chapters/02_in_context_reasoning/results/benchmark_summary_matrix.csv): Metric comparison matrix across all 6 paradigms.

---

## 🛠️ Hardware & Environment Configuration
- **Model**: `meta-llama/Meta-Llama-3-8B-Instruct`
- **Precision**: 4-bit NormalFloat4 (NF4) with double quantization via `bitsandbytes`
- **Compute Dtype**: `torch.float16`
- **Runtime**: Google Colab Free Tier (NVIDIA Tesla T4 GPU, 15.8 GB VRAM, CUDA 12.x)
- **Allocated Model Footprint**: ~5.35 GB VRAM
