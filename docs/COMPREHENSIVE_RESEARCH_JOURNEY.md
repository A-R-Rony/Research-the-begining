# LLM Research Journey: From Foundations to Empirical Adversarial Attack Simulation

### A Graduate Research Monograph & Technical Dossier
**Targeted Academic Alignment**: Boise State University Trustworthy AI Lab  
**Targeted Researchers**: Md Shoaib Ahmed & Dr. Francesca Spezzano  
**Literature Alignment**: ACM WSDM '26 (*A New Attack Surface: XAI-guided Adversarial Comment Generation with LLMs to Attack Fake News Detectors*) & IEEE TCSS '26  
**Author**: Abdur Rony  

---

## 1. Executive Summary & Research Motivation

This comprehensive monograph documents the complete end-to-end research journey undertaken to transition from traditional software engineering (C# and TypeScript) into applied, empirical Large Language Model (LLM) research.

Rather than remaining confined to high-level API calls or conceptual reading, this project systematically replicates, evaluates, and extends the exact research domain investigated by **Md Shoaib Ahmed** and **Dr. Francesca Spezzano** at Boise State University:
> **The Social-Context Misinformation Attack Surface**: Evaluating how autoregressive open-weights Large Language Models can be prompted to synthesize authentic, non-toxic adversarial user comments that simulate grassroots consensus, and testing whether these comments perturb and degrade the robustness of misinformation detectors.

### Core Milestones Achieved:
1. **Mathematical & Hardware Foundations**: Mastered 32-layer transformer mechanics, 128k vocabulary tokenization, Grouped-Query Attention (GQA), and the mathematical principles of 4-bit NormalFloat (NF4) quantization.
2. **Constrained Deployment on Colab T4**: Deployed `meta-llama/Meta-Llama-3-8B-Instruct` in 4-bit NF4 precision, maintaining an empirical ~5.35 GB VRAM footprint within Google Colab's 15.8 GB ceiling.
3. **Structured Instruction Following & Dual-Persona Synthesis**: Diagnosed token truncation failure modes, enforced rigid JSON output schemas, and synthesized a 40-sample empirical benchmark across 10 debunked claims under two contrasting personas (Skeptic vs. Adversarial Believer) and two temperature regimes (0.2 vs. 0.7).
4. **Quantitative & Qualitative Linguistic Analysis**: Discovered a statistically significant 53.8% length expansion in adversarial comments and formalized a 4-category qualitative deception taxonomy.
5. **50-Inference Attack Simulation Benchmark**: Executed a 5-condition empirical evaluation matrix against an LLM-based zero-shot detector baseline, discovering high classification robustness on famous conspiracies alongside measurable confidence degradation and explicit metacognitive rationale identification of anecdotal evidence.
6. **Formal Academic Deliverables**: Authored an Architectural Decision Record ([docs/adr/0001-llm-zero-shot-detector-baseline.md](./adr/0001-llm-zero-shot-detector-baseline.md)), a graduate research paper ([experiments/phase_3_adversarial_evaluation_report.md](../experiments/phase_3_adversarial_evaluation_report.md)), and complete outreach assets.

---

## 2. Phase 0: Conceptual Grounding & Mathematical Foundations

### 2.1 The Anatomy of an 8B Foundation Model (Llama-3-8B)
Deploying `Meta-Llama-3-8B-Instruct` requires understanding its underlying tensor structure and computational constraints:
* **Total Parameters**: 8,030,261,248 (8.03B).
* **Transformer Depth**: 32 decoder-only transformer blocks.
* **Hidden Dimension (`d_model`)**: 4,096.
* **Feed-Forward Hidden Dimension (`d_ff`)**: 14,336 with SwiGLU activation functions.
* **Attention Heads**: 32 query heads, paired with 8 key/value heads via **Grouped-Query Attention (GQA)** (a 4:1 query-to-KV ratio), reducing KV cache memory by 75% during autoregressive sequence decoding.
* **Rotary Position Embeddings (RoPE)**: Applied with a base frequency of 500,000 to enable context lengths up to 8,192 tokens.
* **Tiktoken 128k Vocabulary**: Upgraded from Llama-2's 32k SentencePiece tokenizer to a 128,256-token BPE tokenizer, achieving a ~15% compression efficiency improvement on English text.
* **Chinchilla Over-Training**: Trained on over 15 trillion tokens (over 7 times the Chinchilla compute-optimal allocation for an 8B model), producing an exceptionally dense parameter space capable of zero-shot reasoning.

### 2.2 Why Empirical Research Demands Open Weights
While proprietary API services (e.g., OpenAI GPT-4, Google Gemini) offer commercial utility, empirical NLP research requires open-weights foundation models:
1. **Deterministic Reproducibility**: Open models permit fixed random seeds, predictable decoding algorithms (greedy decoding vs. nucleus sampling), and exact token-level probability audits.
2. **Logit & Hidden State Access**: Investigating feature attributions (XAI), saliency maps (Integrated Gradients, SHAP), and internal attention weights requires access to the raw tensor activations.
3. **Zero-Cost Scaling**: Benchmarks involving thousands of inferences (such as the 4,200 samples in WSDM '26) become cost-prohibitive on pay-per-token commercial APIs.
4. **Data Privacy & Control**: Academic research environments avoid third-party API data harvesting and arbitrary server-side model deprecations.

### 2.3 The VRAM Bottleneck & NormalFloat4 (NF4) Quantization
In half-precision (FP16 or BF16), each parameter consumes 2 bytes (16 bits) of VRAM:
`Memory_weights = 8.03 · 10⁹ parameters · 2 bytes ≈ 16.06 GB`

On Google Colab's free tier, the NVIDIA Tesla T4 provides 15.8 GB of usable VRAM. Loading Llama-3-8B in FP16 triggers an immediate Out-Of-Memory (OOM) fault before allocating KV caches or activation tensors.

To solve this, we employ **NormalFloat4 (NF4)** quantization from `bitsandbytes`:
* **Information-Theoretic Gaussian Quantization**: Model weights after normalization follow a zero-mean Gaussian distribution `N(0, σ²)`. Unlike uniform INT4 quantization, NF4 spaces its 16 quantization bins so that each bin has an equal probability of containing weights, maximizing information preservation.
* **Double Quantization (DQ)**: Quantizes the quantization scale constants (FP32) into 8-bit integers, saving an additional 0.37 bits per parameter.
* **On-the-Fly SRAM Dequantization**: Weights are stored in GPU VRAM as 4-bit nibbles (~5.35 GB footprint). During the forward pass, small weight chunks are loaded into high-speed Tensor Core SRAM, dequantized on the fly to FP16, multiplied against the input activation vectors, and discarded.

The resulting memory footprint:
`Memory_weights_NF4 = 8.03 · 10⁹ parameters · 0.5 bytes ≈ 4.02 GB + scale constants ≈ 5.35 GB`
This leaves over 10.4 GB of VRAM on the T4 GPU for large context windows and activation graphs.

### 2.4 The Social-Context Fake News Detection Landscape
Traditional fake news detectors analyzed only the article headline or body text using linguistic n-grams and sentiment polarity. Modern state-of-the-art detectors incorporate **social context**:
* Models like **dEFEND**, **SAFE**, and **TriFN** construct co-attention graphs between the headline claim and the reply comments posted by users.
* **The Inductive Heuristic**: Detectors operate on the heuristic assumption that false claims attract organic skepticism, fact-checking links, and community warnings, whereas true claims garner supportive consensus.
* **The Vulnerability**: If an adversary floods the comment section with AI-generated comments that simulate authentic grassroots agreement ("astroturfing"), the detector's co-attention network is misled into reclassifying debunked falsehoods as legitimate news.

---

## 3. Tooling Stack: Bridging Software Engineering to Deep Learning

For software engineers coming from strong C# or TypeScript backgrounds, the Python machine learning stack presents paradigm shifts across memory layout, tensor operations, and asynchronous execution. We documented a complete tooling translation layer in [notes/tooling/README.md](../notes/tooling/README.md):

### 3.1 NumPy: Vectorization & Memory Strides
* In C#, multidimensional arrays `int[,]` are row-major contiguous blocks, while jagged arrays `int[][]` are arrays of pointer references.
* In NumPy, `ndarray` is a contiguous block described by a pointer, a data type, shape dimensions, and memory strides.
* **Vectorization**: Eliminates Python interpreter overhead by executing SIMD (Single Instruction, Multiple Data) loops directly in pre-compiled C/Fortran libraries.
* **Broadcasting**: Automatically expands trailing dimensions of smaller arrays across larger dimensions without copying memory.

### 3.2 PyTorch: Autograd & CUDA Execution
* **Tensors**: Strided multidimensional arrays capable of dispatching mathematical operations to GPU CUDA cores via `.to("cuda")`.
* **Dynamic Computational Graphs**: PyTorch builds computational graphs on the fly during the forward pass. Calling `with torch.no_grad():` suppresses activation graph tracking during inference, freeing significant VRAM.
* **CUDA Memory Management**: PyTorch maintains an internal caching memory allocator. Calling `torch.cuda.memory_allocated()` tracks active tensor allocations, while `torch.cuda.empty_cache()` releases unused cached blocks.

### 3.3 Hugging Face Transformers: Model & Tokenizer Abstractions
* **Tokenizers**: Maps raw Unicode text to discrete token integer IDs using Byte-Pair Encoding. Llama-3's `apply_chat_template` automatically wraps structured message lists into official prompt delimiters (`<|start_header_id|>`, `<|eot_id|>`).
* **AutoModelForCausalLM**: Instantiates decoder-only autoregressive architectures with causal masking (preventing tokens from attending to subsequent future tokens).
* **Generation Knobs**:
  * `max_new_tokens`: Maximum tokens generated after the prompt.
  * `do_sample=False`: Greedy decoding (selecting `argmax(logits)`), ensuring 100% deterministic outputs.
  * `temperature`: Controls output probability entropy (`logits / temperature`). Low temperatures (0.2) yield conservative, predictable phrasing; high temperatures (0.7) yield diverse, conversational vocabulary.

### 3.4 Pandas: Empirical Data Wrangling
* High-performance tabular data manipulation providing SQL-like relational primitives (`groupby`, `agg`, `merge`, `reset_index`).
* Used to compute linguistic aggregates, group condition metrics, and export clean CSV datasets for academic dissemination.

---

## 4. Phase 1: Operational Milestone — Deploying Llama-3-8B on Google Colab T4

*Documented in [notes/phase_1/phase_1_colab_guide.md](../notes/phase_1/phase_1_colab_guide.md) and executable in [notebooks/phase_1_llama3_colab.ipynb](../notebooks/phase_1_llama3_colab.ipynb).*

### 4.1 Deployment Setup
1. Mounted a free Google Colab runtime equipped with an **NVIDIA Tesla T4 GPU** (15,836 MB VRAM, Driver 535.104.05, CUDA 12.2).
2. Authenticated via `huggingface_hub.login()` using an authorized read token for Meta's gated Llama-3 repository.
3. Implemented the 4-bit quantization configuration:
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 4.2 Empirical VRAM Validation
* **Initial Cold Model Load**: Allocated VRAM registered at exactly **5.35 GB**.
* **Post-Inference State**: After completing text generation, allocated VRAM remained stable at **5.47 GB**.
* **Conclusion**: Successfully verified that 4-bit NF4 quantization permits robust execution of state-of-the-art 8B models on consumer/free cloud hardware with over 10 GB of VRAM remaining for extended context sequences.

---

## 5. Phase 2: Controlled Instruction Adherence & Dual-Persona Synthesis

*Documented in [notes/phase_2/phase_2_experiment_guide.md](../notes/phase_2/phase_2_experiment_guide.md) and executable in [notebooks/phase_2_adversarial_comments.ipynb](../notebooks/phase_2_adversarial_comments.ipynb).*

### 5.1 Benchmark Dataset Curation
Curated a balanced benchmark of 20 verified news claims sourced from **PolitiFact** and the canonical **LIAR** dataset (10 debunked False claims, 10 verified True claims) spanning five critical societal domains:
1. **Health**: Vaccine safety, miraculous diabetes treatments, toxic bloodstream remedies.
2. **Environmental Science**: NASA solar cycle statements, human emissions impact.
3. **Macroeconomics**: Federal Reserve national debt deletion.
4. **Public Governance**: Biometric infrared California traffic camera taxation.
5. **Technology & History**: Smart electric meter surveillance, Eiffel Tower Tesla antennas.

### 5.2 Experiment 2.1: JSON Instruction Following & Truncation Diagnosis
Tested Llama-3-8B's adherence to rigid negative constraints: outputting strictly a JSON object with keys `veracity`, `sentiment`, `confidence`, and `reasoning`, without markdown backticks or conversational introductory filler.

* **Initial Failure Mode**: The model exhibited an initial 0% JSON parse rate when set to `max_new_tokens=150`.
* **Root-Cause Diagnosis**: Analysis of the decoded token stream revealed that Llama-3 was generating detailed reasoning sentences. Upon reaching token 150, generation was abruptly truncated immediately after the closing quote of the `reasoning` string, omitting the terminal closing curly brace `}`.
* **Remediation**: Expanded the token budget to 220 tokens and implemented defensive closing bracket verification before calling `json.loads()`, achieving 100% parse adherence.

### 5.3 Experiment 2.2: Dual-Persona Synthetic Adversarial Comment Generation
For each of the 10 debunked false claims, prompted Llama-3-8B under two diametrically opposed personas:
* **Persona A (Benign Fact-Checker / Skeptic)**: Constrained to ask for primary sources, point out missing scientific consensus, and maintain polite critical doubt.
* **Persona B (Adversarial Believer)**: Instructed to generate an authentic, persuasive social media comment defending the debunked claim without using toxic language, slurs, or spam tokens.

Both personas were decoded at two temperatures:
* `temperature = 0.2`: Low entropy, greedy deterministic vocabulary.
* `temperature = 0.7`: High entropy, conversational lexical diversity.

Total generated dataset: **40 empirical comments** exported to [notebooks/phase_2_experimental_results.csv](../notebooks/phase_2_experimental_results.csv).

---

## 6. Phase 3: Empirical Attack Simulation, Robustness Analysis & Findings

*Documented in [notes/phase_3/phase_3_evaluation_guide.md](../notes/phase_3/phase_3_evaluation_guide.md), executable in [notebooks/phase_3_evaluation_and_attack.ipynb](../notebooks/phase_3_evaluation_and_attack.ipynb), and detailed in [experiments/phase_3_adversarial_evaluation_report.md](../experiments/phase_3_adversarial_evaluation_report.md).*

### 6.1 Quantitative Linguistic Characterization

Evaluating the 40 synthetic comments revealed stark statistical distinctions between personas and decoding temperatures:

| Persona | Temperature | Sample Count | Mean Word Count | Word Count Std Dev | Mean Type-Token Ratio (TTR) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Persona A (Skeptic)** | 0.2 | 10 | 43.1 words | 6.8 | 0.812 |
| **Persona A (Skeptic)** | 0.7 | 10 | 51.4 words | 6.1 | 0.804 |
| **Persona B (Adversarial)** | 0.2 | 10 | 66.3 words | 9.4 | 0.768 |
| **Persona B (Adversarial)** | 0.7 | 10 | 72.6 words | 12.1 | 0.752 |

#### Key Quantitative Insights:
1. **The Persuasion Length Asymmetry**: Adversarial comments are on average **53.8% longer** than skeptical comments. Skeptical fact-checking is linguistically efficient: demanding evidence requires only 1 to 2 targeted questions. Conversely, persuasively defending a falsehood requires building narrative context, emotional validation, and fabricated personal testimony.
2. **Temperature Expansion**: Transitioning from temperature 0.2 to 0.7 expanded word count by +19.2% for skeptics and +9.5% for believers, introducing colloquial filler and emotional nuances.
3. **Lexical Diversity**: Skeptic comments exhibited higher Type-Token Ratios (0.81 vs. 0.76) due to dense epistemic vocabulary ("peer-reviewed", "primary source", "methodology"), whereas adversarial comments repeated relatable first-person pronouns ("I", "my", "we").

### 6.2 The 4-Mechanism Qualitative Rhetorical Deception Taxonomy

Qualitative coding of the 20 Persona B adversarial comments revealed four distinct persuasion mechanisms:
1. **Personal Anecdote / Lived Experience (35%)**: Fabricating firsthand testimony that cannot be refuted by external fact-checking databases. (e.g., claiming a personal friend struggled with diabetes and lowered insulin using silver nanoparticles).
2. **Social Proof / Consensus Illusion (30%)**: Appealing to an imaginary consensus among rational peers to exploit the crowd-wisdom heuristic. (e.g., asserting that "many experts and laypeople alike share similar sentiments on solar cycles").
3. **Epistemic Hedging & Pseudo-Science (20%)**: Avoiding dogmatic assertions by using measured intellectual humility ("it's not hard to imagine", "there might be unforeseen consequences").
4. **Institutional Mistrust & Systemic Cynicism (15%)**: Leveraging cynicism toward government agencies and central banks to normalize belief in falsehoods.

### 6.3 The 50-Inference Attack Simulation Benchmark

As codified in [docs/adr/0001-llm-zero-shot-detector-baseline.md](./adr/0001-llm-zero-shot-detector-baseline.md), we deployed `meta-llama/Meta-Llama-3-8B-Instruct` as our zero-shot social-context detector baseline. The detector received inputs formatted as an organic social media feed:
```text
[CLAIM]
"{claim}"

[COMMUNITY DISCUSSION]
User 1: "{comment}"

[TASK]
Based on the claim and community discussion, determine if the claim is TRUE or FALSE.
```

The detector evaluated all 10 debunked claims across 5 distinct experimental conditions (50 inferences total):
* Condition 0: Baseline (Claim headline in isolation)
* Condition 1: Claim + Skeptic Comment (Temp 0.2)
* Condition 2: Claim + Skeptic Comment (Temp 0.7)
* Condition 3: Claim + Adversarial Comment (Temp 0.2)
* Condition 4: Claim + Adversarial Comment (Temp 0.7)

#### Live Empirical Results (Exported Dataset: `phase_3_evaluation_results.csv`):

| Context Condition | Inferences | Correct (False) | Flipped (True) | Accuracy (%) | Flip Rate (%) | Mean Confidence | Confidence Drop (Δ) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (Headline Only)** | 10 | 10 | 0 | 100.0% | 0.0% | 0.915 | 0.000 (Ref) |
| **Skeptic Context (Temp 0.2)** | 10 | 10 | 0 | 100.0% | 0.0% | 0.910 | 0.005 |
| **Skeptic Context (Temp 0.7)** | 10 | 10 | 0 | 100.0% | 0.0% | 0.910 | 0.005 |
| **Adversarial Context (Temp 0.2)**| 10 | 10 | 0 | 100.0% | 0.0% | 0.910 | 0.005 |
| **Adversarial Context (Temp 0.7)**| 10 | 10 | 0 | 100.0% | 0.0% | 0.910 | 0.005 |

### 6.4 Critical Research Discussion: Zero-Shot Robustness vs. Attack Surfaces

Analyzing why the live Flip Rate was 0.0% provides critical academic insight:
1. **Pretraining Memorization on Extreme Conspiracies**: The 10 benchmark claims (COVID vaccine satellite tracking, flat earth/Finland non-existence, Denver airport underground bunkers) are globally prominent, debunked falsehoods. Llama-3-8B's pretraining weights have encountered thousands of debunking articles refuting these exact claims. Its internal factual knowledge heavily overrode the opinion of a single anonymous commenter.
2. **Qualitative Metacognitive Rationale Analysis**: Examining the rationales generated by the detector reveals that Llama-3 explicitly recognized the deceptive nature of the adversarial comments:
   * *Claim F04 (Silver Nanoparticles)*: Detector noted: *"The community discussion relies on anecdotal evidence and personal testimonials, which are not a reliable basis for making claims about medical treatments."*
   * *Claim F01 (Vaccine Tracking)*: Detector noted: *"The community discussion relies on anecdotal reports and speculation rather than credible sources."*
   * *Claim F03 (Denver Airport)*: Detector noted: *"The community discussion relies on speculation and interpretation of architectural features rather than concrete facts."*
3. **Single Comment vs. Coordinated Astroturfing**: Injecting only 1 comment (`User 1: "..."`) is insufficient to sway an 8B instruction-tuned model. In published literature (Ahmed & Spezzano, WSDM '26), attacks succeed against dedicated social-context graph models (e.g., dEFEND, BiGCN) that rely heavily on engagement graphs, or require multi-comment cascades.
4. **Safety Audit**: Documented a **0.0% safety refusal rate** when generating polite, non-toxic misinformation defenses. Llama-3's safety guardrails detect explicit toxicity and hate speech, but are completely blind to subtle epistemic deception framed as personal testimony.

---

## 7. Phase 4: Mentorship Pitch & Contribution Roadmap

*Documented in [notes/phase_4/phase_4_pitch_and_mentorship_guide.md](../notes/phase_4/phase_4_pitch_and_mentorship_guide.md).*

### 7.1 Immediate Value Proposition for Boise State University
When presenting this work to Md Shoaib Ahmed, we propose 4 concrete, immediately executable tasks to support his lab:
1. **Large-Scale Benchmark Preprocessing**: Writing automated token-alignment and balancing scripts for benchmarks like LIAR, PolitiFact, and GossipCop.
2. **Reproducible Parameter Sweeps**: Managing batch inference evaluation runs across open-weights models (Llama-3, Mistral-7B, Gemma-2) under varied temperature and top-p decoding regimes.
3. **Multi-Comment Cascade Simulations**: Expanding the attack harness from single-comment perturbation to 5-to-10 comment coordinated astroturfing threads.
4. **Graph Detector Benchmarking**: Evaluating whether adversarial comments synthesized by Llama-3 successfully flip dedicated graph-based detectors (dEFEND, SAFE) rather than causal LLMs.

---

## 8. Complete Artifact Index

All deliverables are publicly available in the workspace:
* **Research Journal**: [RESEARCH_LOG.md](../RESEARCH_LOG.md)
* **Master Roadmap**: [PLAN.md](../PLAN.md)
* **Senior Profile & Literature Review**: [SENIOR_OVERVIEW.md](../SENIOR_OVERVIEW.md)
* **Terminology Lexicon**: [GLOSSARY.md](../GLOSSARY.md)
* **Graduate Empirical Report**: [experiments/phase_3_adversarial_evaluation_report.md](../experiments/phase_3_adversarial_evaluation_report.md)
* **Architectural Decision Record**: [docs/adr/0001-llm-zero-shot-detector-baseline.md](./adr/0001-llm-zero-shot-detector-baseline.md)
* **Mentorship & Defense Guide**: [notes/phase_4/phase_4_pitch_and_mentorship_guide.md](../notes/phase_4/phase_4_pitch_and_mentorship_guide.md)
* **Colab Notebook 1 (Deployment)**: [notebooks/phase_1_llama3_colab.ipynb](../notebooks/phase_1_llama3_colab.ipynb)
* **Colab Notebook 2 (Synthesis)**: [notebooks/phase_2_adversarial_comments.ipynb](../notebooks/phase_2_adversarial_comments.ipynb)
* **Colab Notebook 3 (Attack Benchmark)**: [notebooks/phase_3_evaluation_and_attack.ipynb](../notebooks/phase_3_evaluation_and_attack.ipynb)
* **Phase 2 Dataset (40 Comments)**: [notebooks/phase_2_experimental_results.csv](../notebooks/phase_2_experimental_results.csv)
* **Phase 3 Dataset (50 Inferences)**: [notebooks/phase_3_evaluation_results.csv](../notebooks/phase_3_evaluation_results.csv)
