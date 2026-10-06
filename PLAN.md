# Master Research & Implementation Plan
### Targeted Alignment: Boise State University Trustworthy AI Lab (Md Shoaib Ahmed / Dr. Francesca Spezzano)

---

## 🎯 The Strategic Objective
Progress from zero ML experience to executing an empirical applied LLM project that directly matches Md Shoaib Ahmed's core research pipeline:
**"Deploying an open-weights 8B model (Llama-3-8B) on Google Colab to evaluate instruction following and generate synthetic user comments (benign vs. adversarial) on news headlines to test detector robustness."**

---

## 📍 Phase 0: Conceptual Grounding (His Exact Technical Vocabulary)
*Before running code, master the terminology used in his lab so you can speak his language:*
- **Topic 0.1: Why 8B Parameters (Llama-3-8B)?**
  - What are weights, parameters, and tokens?
  - Why his lab avoids closed APIs (GPT-4) for generating 4,200+ samples: cost, lack of reproducible logits/seeds, and API rate limits.
- **Topic 0.2: Hardware & VRAM Bottlenecks (Google Colab T4)**
  - Why an 8B model needs ~16 GB VRAM in standard FP16 precision.
  - How **4-bit NormalFloat Quantization (`bitsandbytes`)** compresses the model down to ~5.5 GB VRAM, fitting safely on a free Colab T4 GPU (16 GB) without OOM crashes.
- **Topic 0.3: The Senior's Core Research Problem (WSDM '26 & TCSS '26)**
  - How modern fake news detectors work: they analyze both the **news headline** AND the **user comments** below it.
  - What an **adversarial comment** is: an AI-generated comment disguised as an ordinary social media response that tricks a fake news classifier into believing a false story.
  - The role of **InstructNet**: classifying and enforcing strict instruction constraints on generated text.

---

## 📍 Phase 1: Operational Milestone — Running Llama-3-8B on Google Colab
*The baseline engineering test he asked you to complete:*
- **Step 1.1**: Open a Google Colab notebook and mount the free **NVIDIA T4 GPU** runtime.
- **Step 1.2**: Verify GPU health with `!nvidia-smi` and configure your Hugging Face authentication token (`huggingface_hub`).
- **Step 1.3**: Load `meta-llama/Meta-Llama-3-8B-Instruct` in 4-bit precision (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`).
- **Step 1.4**: Write a modular generation function utilizing Llama-3's official chat template (`apply_chat_template`).
- **Step 1.5**: Track GPU VRAM before and after generation (`torch.cuda.memory_allocated()`).

---

## 📍 Phase 2: The Aligned Research Experiment (The Senior's Problem Space)
*Conducting the exact experiment that will impress him:*
- **Dataset Selection**: Use a small subset (e.g. 20–50 samples) from a standard open fake news benchmark (e.g. **LIAR** or **PolitiFact** via Hugging Face `datasets`).
- **Experiment 2.1: Controlled Instruction Following (InstructNet theme)**
  - Prompt Llama-3-8B to classify the claim and output strictly formatted JSON metadata (claim veracity, sentiment, confidence).
  - Calculate format adherence rate.
- **Experiment 2.2: Synthetic Adversarial Comment Generation (WSDM '26 theme)**
  - For each false news headline, prompt Llama-3-8B under two distinct personas:
    1. **Persona A (Skeptical / Benign Fact-Checker)**: Generates a comment questioning the source and requesting evidence.
    2. **Persona B (Adversarial / Persuasive Believer)**: Generates a subtle, convincing comment defending the false claim without using overt spammy words.
  - Test temperature variations (`temperature=0.2` vs `temperature=0.7`) to see diversity vs. consistency.

---

## 📍 Phase 3: Research Evaluation & Reporting
*Packaging the experiment like a graduate research assistant:*
- Log the generated comments and metrics into a clear markdown report: `experiments/adversarial_comment_generation.md`.
- Analyze the linguistic features: Did the adversarial comments use authoritative tone, personal anecdotes, or emotional appeal?
- Highlight failure modes (e.g., cases where Llama-3's safety guardrails refused to generate deceptive comments).

---

## 📍 Phase 4: Pitching to Your Senior
*The high-impact message to send him:*
- Share your clean GitHub repository and runnable Colab notebook.
- Message script:
  > *"Hey Shoaib bhai, I followed your advice and set up an 8B model (Meta-Llama-3-8B-Instruct) on Google Colab using 4-bit NF4 quantization on a T4 GPU. Knowing your work on WSDM '26 regarding adversarial comments against fake news detectors, I designed an experiment using open news headlines from the LIAR benchmark to generate benign vs. adversarial comments under controlled personas and evaluated Llama-3's adherence and safety behavior. I documented the pipeline and results in my repository: https://github.com/A-R-Rony/Research-the-begining. I'd love to show you the notebook and see if I can assist with dataset preprocessing or model evaluations in your lab."*

---

## 🧭 Reference Documents
- **[CONTEXT.md](./CONTEXT.md)**: Research background, mentorship goal, and log update protocol.
- **[RESEARCH_LOG.md](./RESEARCH_LOG.md)**: Living daily research journal.
- **[SENIOR_OVERVIEW.md](./SENIOR_OVERVIEW.md)**: Md Shoaib Ahmed's complete profile and publication analysis.
