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

### 5. Open Questions & Next Actions for Day 2
- [ ] Create a Hugging Face account and request access to the gated `meta-llama/Meta-Llama-3-8B-Instruct` model.
- [ ] Open a fresh Google Colab notebook and run the initial GPU verification (`!nvidia-smi`).
- [ ] Write the Python script to load the model in 4-bit and run my first test generation.
