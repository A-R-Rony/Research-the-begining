# Chapter 2: Open-Weight In-Context Reasoning & Lightweight RAG

Welcome to Chapter 2 of the LLM research journey. Building directly upon our operational foundation with 4-bit quantized inference on `meta-llama/Meta-Llama-3-8B-Instruct` and adversarial comment generation, this chapter explores **in-context reasoning paradigms** from mechanistic first principles to empirical evaluation.

---

## Pedagogical & Research Objectives

1. **Master Open-Weight Prompt Serialization**: Learn how instruction-tuned open-weight models process system prompts, chat templates, and special tokens (`<|start_header_id|>`, `<|eot_id|>`), alongside the memory scaling laws of the KV cache.
2. **Evaluate In-Context Learning (ICL)**: Understand how k demonstration exemplars guide the model's activation trajectories without gradient updates, comparing native chat-template conversation turns against monolithic prompt packing.
3. **Implement Grounding & In-Memory RAG**: Eliminate parametric hallucinations by conditioning generation on external evidence passages using bi-encoder vector embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and pure PyTorch cosine similarity.
4. **Enforce Structured Chain-of-Thought (CoT)**: Implement three-stage epistemic reasoning decomposition to eliminate rambling and narrative drift in 8B models, verifying premises before reaching terminal labels.
5. **Conduct Multi-Metric Benchmarking**: Evaluate Zero-Shot, Few-Shot, Context-Grounded, Dynamic RAG, and CoT side-by-side across a 20-claim benchmark suite measuring accuracy, token consumption, latency, format adherence, and context faithfulness.

---

## Governing Documents & Architecture

- **Specification**: [0001-open-weight-in-context-reasoning-spec.md](../../docs/specs/0001-open-weight-in-context-reasoning-spec.md)
- **Architectural Decision Record**: [0002-in-context-reasoning-and-lightweight-rag.md](../../docs/adr/0002-in-context-reasoning-and-lightweight-rag.md)
- **Domain Terminology**: [GLOSSARY.md](../../GLOSSARY.md)
- **Prior Foundation**: [COMPREHENSIVE_RESEARCH_JOURNEY.md](../../docs/COMPREHENSIVE_RESEARCH_JOURNEY.md) (Phases 0 through 4)

---

## Chapter Directory Layout

```text
chapters/02_in_context_reasoning/
├── README.md                                      # This syllabus and navigation index
│
├── notes/                                         # Theory and learning curriculum (read before coding)
│   ├── topic_2.1_open_weight_prompt_mechanics.md  # Tokenization, chat templates, KV cache VRAM math
│   ├── topic_2.2_zero_shot_and_few_shot_icl.md    # Parametric boundaries, ICL theory, native vs monolithic turns
│   ├── topic_2.3_context_grounding_and_rag.md     # Grounding faithfulness, dense embeddings, PyTorch cosine search
│   └── topic_2.4_chain_of_thought_reasoning.md    # Epistemic decomposition, drift prevention, self-consistency
│
├── deep_dives/                                    # Mechanistic deep dives & representation engineering
│   └── deep_dive_activation_steering.md           # Activation steering, SAEs, & Eiffel Tower Llama mechanics
│
├── notebooks/                                     # Runnable Google Colab T4 environments
│   ├── chapter_2_in_context_reasoning.ipynb       # Interactive lab executing all four paradigms
│   └── activation_steering_llama3_1.ipynb         # Hands-on activation steering & sweep comparison on Llama 3.1
│
├── data/                                          # Benchmark datasets and knowledge corpora
│   ├── benchmark_claims_20.json                   # 20 curated claims (5 baseline + 15 nuanced context claims)
│   ├── knowledge_corpus.json                      # Reference evidence passages for semantic retrieval
│   └── exemplars_bank.json                        # Formatted exemplars for few-shot prompt compilation
│
└── results/                                       # Empirical benchmarking outputs
    ├── benchmark_results.csv                      # Recorded metrics across all conditions
    └── chapter_2_evaluation_report.md             # Formal comparative evaluation report
```

---

## Study & Execution Roadmap

### Stage 1: Foundational Theory (Notes)
- Study [notes/topic_2.1_open_weight_prompt_mechanics.md](./notes/topic_2.1_open_weight_prompt_mechanics.md) to understand how strings are mapped to tokens and wrapped by Llama-3 special delimiters.
- Study [notes/topic_2.2_zero_shot_and_few_shot_icl.md](./notes/topic_2.2_zero_shot_and_few_shot_icl.md) to master how few-shot exemplars shape attention matrices.
- Study [deep_dives/deep_dive_activation_steering.md](./deep_dives/deep_dive_activation_steering.md) for a mechanistic deep dive into direct latent intervention, Sparse Autoencoders (SAEs), and the "Eiffel Tower Llama" experiment.
- Study [notes/topic_2.3_context_grounding_and_rag.md](./notes/topic_2.3_context_grounding_and_rag.md) to understand semantic vector spaces and dense retrieval mechanics.
- Study [notes/topic_2.4_chain_of_thought_reasoning.md](./notes/topic_2.4_chain_of_thought_reasoning.md) to explore multi-step deductive reasoning and prompt constraints.

### Stage 2: Data Preparation (Data)
- Review the 20 benchmark claims in [data/benchmark_claims_20.json](./data/benchmark_claims_20.json).
- Review companion evidence passages in [data/knowledge_corpus.json](./data/knowledge_corpus.json).

### Stage 3: Interactive Execution (Notebooks)
- Open and run [notebooks/chapter_2_in_context_reasoning.ipynb](./notebooks/chapter_2_in_context_reasoning.ipynb) on Google Colab with an NVIDIA T4 GPU.
- Execute Zero-Shot, Few-Shot (testing k=1, 3, 5 with template comparison), Context-Aware Grounding + RAG, and Structured CoT.
- Open and run [notebooks/activation_steering_llama3_1.ipynb](./notebooks/activation_steering_llama3_1.ipynb) to experiment hands-on with Contrastive Activation Addition (CAA), SAE steering, and intensity sweeps on Llama 3.1.

### Stage 4: Empirical Evaluation (Results)
- Inspect generated metrics in [results/benchmark_results.csv](./results/benchmark_results.csv).
- Read the findings and error analysis in [results/chapter_2_evaluation_report.md](./results/chapter_2_evaluation_report.md).
