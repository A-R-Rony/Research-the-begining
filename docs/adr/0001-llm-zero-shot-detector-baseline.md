# 0001. Adopt LLM Zero-Shot Verifier as Social-Context Detector Baseline

## Context
In fake news detection literature (e.g., dEFEND, TriFN, and TCSS '26 benchmarks), social-context detectors often employ specialized Graph Neural Networks (GNNs) or heavy fine-tuned sequence encoders to co-attend over news articles and user comment threads. However, training and evaluating custom GNN architectures requires complex graph preprocessing pipelines and significant training compute that exceeds standard Google Colab T4 resource allocations.

## Decision
We adopted `meta-llama/Meta-Llama-3-8B-Instruct` loaded in 4-bit NF4 precision under a structured zero-shot verification prompt as our baseline social-context detector. The model evaluates claims under three distinct conditions: isolated headlines, headlines paired with benign skeptical comments, and headlines paired with synthetic adversarial comments.

## Consequences
- **Positive**: Enables zero-setup, reproducible inference directly within the existing Google Colab T4 environment without additional model checkpoints or out-of-memory risks.
- **Positive**: Directly measures how modern autoregressive LLM reasoning and confidence scores are swayed by synthetic social consensus.
- **Trade-off**: Evaluates generative instruction-following veracity assessment rather than a dedicated discriminative graph-based social network topology.
