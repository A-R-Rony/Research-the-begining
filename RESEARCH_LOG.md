# Daily Research Journal & Progress Log

> **Project**: Applied LLM Research: The Beginning  
> **Target Mentorship**: Md Shoaib Ahmed (Ph.D. Researcher, Boise State University / Trustworthy AI Lab)  
> **Purpose**: A living daily log of what I am learning, building, and conceptualizing. This document is updated day-by-day based on real progress and understanding.

---

## 📅 Day 1: October 6, 2026 — Orientation, Alignment & Conceptual Foundation

### 1. What I Set Out To Do Today
- Clarify how to break into **Applied LLM Research** without prior AI/ML experience.
- Understand the initial hints from my senior researcher:
  - *"Try to run a model like Llama with 8B parameters in Google Colab / Kaggle."*
- Unpack what his research lab actually does so my efforts are laser-focused on problems he cares about.
- Set up a clean, structured research repository and learning roadmap.

---

### 2. Key Discoveries About My Senior & His Lab
I investigated the background and publications of my senior, **Md Shoaib Ahmed** (Ph.D. student at Boise State University, advised by Dr. Francesca Spezzano):
- **Core Research Areas**: Trustworthy & Responsible AI, Adversarial LLMs, Fake News & Misinformation Detection, and Instruction Classification.
- **Key Publication (WSDM '26)**: *"A New Attack Surface: XAI-guided Adversarial Comment Generation with LLMs to Attack Fake News Detectors"*.
- **The Core Insight**:
  - His lab studies how LLMs can generate realistic, persuasive user comments to deceive automated fake news detectors.
  - To generate thousands of experimental comments, researchers cannot rely on expensive closed APIs (like GPT-4). They rely on **open-weights 8B models (like Llama-3-8B)** running on GPUs.
  - **Why he gave me the Colab/8B hint**: He wanted to see if I could independently handle the fundamental operational building blocks of his research—loading an open-weights model on a constrained GPU, prompting it systematically, and observing its outputs.

---

### 3. Core Concepts I Learned Today

#### A. What are "Parameters" and why "8B"?
- **Parameters (Weights)**: The billions of numerical values inside a neural network learned during training that govern how it maps input tokens to output predictions.
- **8B Scale**: 8 Billion parameters. It is considered the modern "sweet spot" in open-source AI: compact enough to fit on accessible/free GPUs, yet powerful enough to follow complex system instructions and reason.

#### B. The GPU & VRAM Constraint (The Math)
- A model's memory footprint is determined by its parameter count and precision:
  `Memory in FP16 (16-bit) ≈ 8 Billion · 2 bytes ≈ 16 GB VRAM`
- A free **Google Colab** instance offers an **NVIDIA T4 GPU** with **16 GB VRAM**.
- Loading an uncompressed 8B model fills the entire 16 GB, leaving zero room for activations, context, or generation—causing an immediate `CUDA Out Of Memory (OOM)` crash.

#### C. What is Quantization (4-bit NF4)?
- Instead of using 16 bits per weight, **Quantization** compresses each weight into **4 bits** (**NormalFloat4** via `bitsandbytes`).
- This drops the model footprint from **~16 GB down to ~5.5 GB**, fitting comfortably on the Colab T4 GPU with plenty of VRAM headroom.

---

### 4. What Was Built in the Repository Today
1. **[CONTEXT.md](./CONTEXT.md)**: Defines my background, research ambitions, and strategic alignment with Shoaib's lab.
2. **[SENIOR_OVERVIEW.md](./SENIOR_OVERVIEW.md)**: Exhaustive profile of Md Shoaib Ahmed, detailing his 767+ citations, h-index, and publication catalog (2023–2026).
3. **[PLAN.md](./PLAN.md)**: A 5-phase master plan bridging fundamentals -> Colab 8B deployment -> synthetic adversarial comment generation -> research report -> mentorship pitch.
4. **[README.md](./README.md)**: Clean index file guiding visitors through the project.

---

### 5. Completed Actions & Transition to Day 2
- [x] Explored and documented foundational linear algebra, tokens, parameters, and 8B architecture.
- [x] Analyzed VRAM constraints, precision formats (FP16 vs NF4), and GPU memory math.

---

## 📅 Day 2: October 8, 2026 — Deep Architecture Audit & Practical Quantization Mechanics

### 1. What I Set Out To Do Today
- Fully complete **Topic 0.1** by mastering Transformer layer mechanics (Self-Attention, Grouped-Query Attention, SwiGLU MLP, and W_O output projection).
- Resolve all mathematical and dimensional questions (why 4096 x 4096 vs 4096 x 1024, head dimension 128, variance scaling 1/√128).
- Streamline **Topic 0.2** into an actionable Python cheat-sheet for `BitsAndBytesConfig` before running Google Colab.

---

### 2. Core Concepts Mastered Today
- **Multi-Head Projection Geometry**:
  - Input token vectors of 4,096 dimensions.
  - Query projection W_Q (4096 x 4096): 32 heads × 128 dimensions.
  - Key/Value projections W_K, W_V (4096 x 1024): 8 heads × 128 dimensions via Grouped-Query Attention (GQA), saving 75% of KV-cache VRAM.
  - Output projection W_O (4096 x 4096): synthesizes clues across all 32 heads into a unified residual update.
- **Attention Mechanics**:
  - Scaling factor 1/√128 stabilizes variance from 128 down to 1.0, preventing vanishing gradients in Softmax.
  - Causal masking (M = -∞ for future tokens) guarantees 0% attention to future tokens.
  - Softmax normalizes scores into probability weights that sum to 1.0.
  - Value aggregation computes the convex combination of semantic vectors.
- **Practical 4-bit Quantization (Topic 0.2)**:
  - `BitsAndBytesConfig` mechanics: `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True`.
  - Core hardware principle: Weights are **stored in 4-bit** in VRAM (~5.5 GB), but **computed in 16-bit** inside Tensor Cores via on-the-fly dequantization.
  - Programmatic VRAM tracking using `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`.

---

### 3. What Was Built in the Repository Today
1. **[topic_0.1_why_8b_parameters.md](./notes/phase_0/topic_0.1_why_8b_parameters.md)**: Exhaustive 680-line theoretical foundation covering tokens, linear algebra, 8B parameter audit, and VRAM memory math.
2. **[topic_0.2_bitsandbytes_vram_cheatsheet.md](./notes/phase_0/topic_0.2_bitsandbytes_vram_cheatsheet.md)**: Production-ready Colab loading script, BitsAndBytes configuration breakdown, and GPU VRAM telemetry helpers.
3. **[AGENTS.md](./AGENTS.md)**: Workspace formatting standard enforcing clean, readable Unicode math notation.

---

### 4. Next Action
- [x] Complete **Topic 0.3**: Deep dive into Md Shoaib Ahmed's core research problem (WSDM '26 & TCSS '26) on multimodal fake news detectors and XAI-guided adversarial comment generation.
- [ ] Transition to **Phase 1**: Execute Google Colab notebook deployment with Llama-3-8B in 4-bit.

---

## 📅 Day 3: October 10, 2026 — Senior's Research Problem (WSDM '26, TCSS '26, InstructNet) & Model Selection

### 1. What Was Tackled Today
- Evaluated model choice between **Llama-3-8B** and **Qwen-2.5 (3B/4B)**:
  - Affirmed Llama-3-8B as the mandatory primary model matching the senior's explicit hardware and parameter challenge.
- Mastered and documented **Topic 0.3**:
  - The architecture of modern social-context fake news detectors (Headline H + User Comments C).
  - The mechanics of adversarial comment attacks (flipping veracity predictions).
  - How Explainable AI (SHAP, LIME, Attention feature attribution) guides LLM generation to bypass classifiers.
  - The role of **InstructNet** in multi-label instruction adherence and automated persona evaluation.

### 2. Concepts Understood
- **Multimodal / Social-Context Detection**:
  - Traditional text-only classifiers fail against journalistic-style fake headlines.
  - Social detectors predict P(Fake | H, C) by aggregating human user reactions beneath the headline. Human skepticism acts as a primary feature.
- **The Adversarial Comment Attack Surface (WSDM '26 & TCSS '26)**:
  - Malicious actors deploy generative LLMs to synthesize authentic-looking comments supporting false claims.
  - Flooding threads with supportive comments shifts classifier probabilities, flipping predictions from Fake to Real.
  - Attack efficiency is quantified using Attack Success Rate (ASR).
- **XAI Guidance in Prompting**:
  - Rather than generic cheerleading, attackers extract high-attribution tokens and tones using XAI tools, then condition LLM system prompts to seamlessly integrate those features.
- **InstructNet Integration**:
  - Classifies instructions across multi-label taxonomies to measure and enforce strict constraint adherence across thousands of synthetic generation samples.

### 3. Artifacts / Code Built
1. **[topic_0.3_adversarial_comments_and_detectors.md](./notes/phase_0/topic_0.3_adversarial_comments_and_detectors.md)**: Exhaustive breakdown of fake news detectors, adversarial comment attacks, XAI-guided prompting, InstructNet, and defenses.
2. **[PLAN.md](./PLAN.md)**: Updated Phase 0 status to 100% completed across Topics 0.1, 0.2, and 0.3.

### 4. Next Day's Focus / Open Questions
- [x] Transition to **Phase 1**: Open Google Colab, verify NVIDIA T4 GPU, authenticate Hugging Face token, and mount `meta-llama/Meta-Llama-3-8B-Instruct` in 4-bit NF4 precision.
- Run baseline prompt inference and record live VRAM consumption.

---

## 📅 Day 4: October 11, 2026 — Phase 1: Operational Milestone — Google Colab T4 Deployment & 4-bit Llama-3-8B

### 1. What I Set Out To Do Today
- Execute **Phase 1** operational milestone: Deploy `meta-llama/Meta-Llama-3-8B-Instruct` on a free Google Colab NVIDIA T4 GPU (15.0 GB VRAM limit).
- Implement 4-bit NF4 double quantization with float16 compute dtype using `bitsandbytes`.
- Build a ready-to-run Jupyter Notebook (`notebooks/phase_1_llama3_colab.ipynb`) and step-by-step execution guide (`notes/phase_1/phase_1_colab_guide.md`).
- Validate GPU hardware health, authenticate Hugging Face tokens, inspect memory footprint (~5.5 GB), and run structured baseline inference via `apply_chat_template`.

### 2. Implementation Artifacts Created
1. **[phase_1_llama3_colab.ipynb](./notebooks/phase_1_llama3_colab.ipynb)**: Executable Colab notebook with 7 structured cells (GPU check, dependencies, HF login, 4-bit model load, VRAM audit, chat template function, smoke test).
2. **[phase_1_colab_guide.md](./notes/phase_1/phase_1_colab_guide.md)**: Companion guide detailing execution instructions, verification criteria, and expected VRAM telemetry.

### 3. Live Execution & Empirical Verification Results ✅
- **Hardware Verified**: NVIDIA Tesla T4 GPU (15.0 GB VRAM) confirmed via `!nvidia-smi`.
- **Hugging Face Authentication**: Gated license validated and weights successfully streamed from `meta-llama/Meta-Llama-3-8B-Instruct`.
- **4-bit NF4 Quantization Loaded**: `BitsAndBytesConfig` successfully mapped model weights onto the GPU without memory overflow.
- **VRAM Telemetry Verified**:
  - Allocated VRAM: ~5.5 GB (weights packed in 4-bit NF4).
  - Free Headroom: > 9.0 GB (abundant capacity for activations, KV cache, and extended context).
- **Inference & Chat Template Validated**:
  - `generate_response` successfully rendered official Llama-3 control tokens via `tokenizer.apply_chat_template`.
  - Smoke test executed in ~1-3 seconds with clean output and prompt token slicing.
  - VRAM remained stable (~5.6-5.8 GB) with zero leaks or OOM issues.

### 4. Milestone Sign-Off & Transition to Phase 2
- [x] **Phase 1 Operational Milestone Achieved**: Proven ability to independently deploy and run open-weights 8B models on constrained GPU hardware.
- [x] **Phase 2 Research Experiment Initialized**: Companion guide and executable notebook built and aligned with WSDM '26 and InstructNet research questions.

---

## 📅 Day 4 (Part II): October 11, 2026 — Phase 2: Design of Adversarial Comment Experiment & Instruction Compliance

### 1. Research Objectives Formulated
- Replicate the experimental setting of Md Shoaib Ahmed's research:
  - **Experiment 2.1 (InstructNet alignment)**: Enforce structured JSON verification constraints on claim evaluation and measure parse compliance rate.
  - **Experiment 2.2 (WSDM '26 alignment)**: Synthesize dual-persona social media comments:
    - **Persona A (Benign Fact-Checker)**: Skeptical, evidence-demanding comments.
    - **Persona B (Adversarial Believer)**: Persuasive, authentic-sounding comments defending false claims without triggering toxicity or spam filters.
  - Temperature study: evaluate deterministic stability (0.2) versus colloquial lexical diversity (0.7).

### 2. Artifacts Built
1. **[phase_2_experiment_guide.md](./notes/phase_2/phase_2_experiment_guide.md)**: Conceptual guide detailing the social-context detector attack surface, JSON schema constraints, persona prompting recipes, and evaluation criteria.
2. **[phase_2_adversarial_comments.ipynb](./notebooks/phase_2_adversarial_comments.ipynb)**: Executable Google Colab research notebook with 8 cells spanning benchmark data ingestion, JSON adherence evaluation, dual-persona generation, and CSV dataset export.

### 3. Immediate Next Step
- Upload `phase_2_adversarial_comments.ipynb` to Google Colab with T4 GPU runtime, execute Cells 1 through 8, and record the experimental metrics and generated comments.

---

## 📅 Day 4 (Part III): October 11, 2026 — Tooling Foundations & Domain Modeling

### 1. Architectural Alignment
- Clarified interactive Jupyter execution mechanics (client-server architecture, persistent IPython Kernel state, global namespace in memory).
- Mapped deep learning research tooling to software engineering concepts for ASP.NET Core (C#) and Angular (TypeScript/JavaScript).

### 2. Artifacts Produced
1. **[GLOSSARY.md](./GLOSSARY.md)**: Canonical domain definitions establishing strict terminology (Tensor, Token, Logits, Attention Matrix, Forward Pass).
2. **[notes/tooling/README.md](./notes/tooling/README.md)**: Index and execution sequence roadmap for the 4-pillar data stack.
3. **[notes/tooling/numpy_essentials.md](./notes/tooling/numpy_essentials.md)**: Contiguous memory buffers, slicing, broadcasting, and cosine similarity.
4. **[notes/tooling/pytorch_for_llms.md](./notes/tooling/pytorch_for_llms.md)**: Device placement (`cuda`), 3D tensor shapes `[Batch, SeqLen, HiddenDim]`, `torch.no_grad()`, and attention matrix extraction.
5. **[notes/tooling/transformers_hf.md](./notes/tooling/transformers_hf.md)**: Tokenizer vs. Model separation, forward pass inspection vs. autoregressive generation.
6. **[notes/tooling/pandas_for_experiments.md](./notes/tooling/pandas_for_experiments.md)**: In-memory table queries, adversarial prompt tracking, and LINQ-equivalent operations.



