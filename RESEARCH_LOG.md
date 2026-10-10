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
  $$\text{Memory in FP16 (16-bit)} \approx 8\text{ Billion} \times 2\text{ bytes} \approx 16\text{ GB VRAM}$$
- A free **Google Colab** instance offers an **NVIDIA T4 GPU** with **16 GB VRAM**.
- Loading an uncompressed 8B model fills the entire 16 GB, leaving zero room for activations, context, or generation—causing an immediate `CUDA Out Of Memory (OOM)` crash.

#### C. What is Quantization (4-bit NF4)?
- Instead of using 16 bits per weight, **Quantization** compresses each weight into **4 bits** (**NormalFloat4** via `bitsandbytes`).
- This drops the model footprint from **~16 GB down to ~5.5 GB**, fitting comfortably on the Colab T4 GPU with plenty of VRAM headroom.

---

### 4. What Was Built in the Repository Today
1. **[CONTEXT.md](./CONTEXT.md)**: Defines my background, research ambitions, and strategic alignment with Shoaib's lab.
2. **[SENIOR_OVERVIEW.md](./SENIOR_OVERVIEW.md)**: Exhaustive profile of Md Shoaib Ahmed, detailing his 767+ citations, h-index, and publication catalog (2023–2026).
3. **[PLAN.md](./PLAN.md)**: A 5-phase master plan bridging fundamentals $\rightarrow$ Colab 8B deployment $\rightarrow$ synthetic adversarial comment generation $\rightarrow$ research report $\rightarrow$ mentorship pitch.
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
- [ ] Complete **Topic 0.3**: Deep dive into Md Shoaib Ahmed's core research problem (WSDM '26 & TCSS '26) on multimodal fake news detectors and XAI-guided adversarial comment generation.
- [ ] Transition to **Phase 1**: Execute Google Colab notebook deployment with Llama-3-8B in 4-bit.

