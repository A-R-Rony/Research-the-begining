# Master Research & Implementation Plan
### Targeted Alignment: Boise State University Trustworthy AI Lab (Md Shoaib Ahmed / Dr. Francesca Spezzano)

---

## 🎯 The Strategic Objective
Progress from zero ML experience to executing an empirical applied LLM project that directly matches Md Shoaib Ahmed's core research pipeline:
**"Deploying an open-weights 8B model (Llama-3-8B) on Google Colab to evaluate instruction following and generate synthetic user comments (benign vs. adversarial) on news headlines to test detector robustness."**

---

## 📍 Phase 0: Conceptual Grounding (His Exact Technical Vocabulary)
*Before running code, master the terminology used in his lab so you can speak his language:*
- **Topic 0.1: Why 8B Parameters (Llama-3-8B)?** ✅ *(Completed)*
  - Documented in: **[topic_0.1_why_8b_parameters.md](./notes/phase_0/topic_0.1_why_8b_parameters.md)**
  - Tokens, vocabularies (tiktoken 128k), linear algebra, 32-layer transformer anatomy, 8.03B parameter audit, GQA, SwiGLU, and Chinchilla over-training.
  - Closed API limitations: why empirical NLP research requires open-weights access for reproducible seeds, logits, and zero-cost scaling (4,200+ samples).
- **Topic 0.2: Hardware & VRAM Bottlenecks & Practical Quantization** ✅ *(Completed)*
  - Documented in: **[topic_0.2_bitsandbytes_vram_cheatsheet.md](./notes/phase_0/topic_0.2_bitsandbytes_vram_cheatsheet.md)**
  - Mathematical VRAM audit (FP32 vs FP16 vs INT8 vs NF4) and Colab T4 15.8 GB ceiling.
  - Python `BitsAndBytesConfig` implementation (`load_in_4bit`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_use_double_quant=True`, `bnb_4bit_compute_dtype=torch.float16`).
  - Mechanics of on-the-fly Tensor Core dequantization and programmatic VRAM tracking (`torch.cuda.memory_allocated()`).
- **Topic 0.3: The Senior's Core Research Problem (WSDM '26 & TCSS '26)** ✅ *(Completed)*
  - Documented in: **[topic_0.3_adversarial_comments_and_detectors.md](./notes/phase_0/topic_0.3_adversarial_comments_and_detectors.md)**
  - How modern fake news detectors work: they analyze both the **news headline** AND the **user comments** below it.
  - What an **adversarial comment** is: an AI-generated comment disguised as an ordinary social media response that tricks a fake news classifier into believing a false story.
  - The role of **InstructNet**: classifying and enforcing strict instruction constraints on generated text.
  - The experimental bridge to Phase 1 and Phase 2.


---

## 📍 Phase 1: Operational Milestone — Running Llama-3-8B on Google Colab ✅ *(Completed)*
*The baseline engineering test he asked you to complete:*
- **Guide**: **[notes/phase_1/phase_1_colab_guide.md](./notes/phase_1/phase_1_colab_guide.md)**
- **Runnable Notebook**: **[notebooks/phase_1_llama3_colab.ipynb](./notebooks/phase_1_llama3_colab.ipynb)**
- **Step 1.1**: Open Google Colab and mount the free **NVIDIA T4 GPU** runtime. ✅
- **Step 1.2**: Verify GPU health with `!nvidia-smi` and configure your Hugging Face authentication token (`huggingface_hub`). ✅
- **Step 1.3**: Load `meta-llama/Meta-Llama-3-8B-Instruct` in 4-bit precision (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`). ✅
- **Step 1.4**: Write a modular generation function utilizing Llama-3's official chat template (`apply_chat_template`). ✅
- **Step 1.5**: Track GPU VRAM before and after generation (`torch.cuda.memory_allocated()`). ✅

---

## 📍 Phase 2: The Aligned Research Experiment (The Senior's Problem Space) ✅ *(Completed)*
*Conducting the exact experiment that will impress him:*
- **Guide**: **[notes/phase_2/phase_2_experiment_guide.md](./notes/phase_2/phase_2_experiment_guide.md)**
- **Runnable Notebook**: **[notebooks/phase_2_adversarial_comments.ipynb](./notebooks/phase_2_adversarial_comments.ipynb)**
- **Dataset Selection**: Curated 20-sample balanced benchmark from **PolitiFact** & **LIAR** (10 False, 10 True). ✅
- **Experiment 2.1: Controlled Instruction Following (InstructNet theme)** ✅
  - Prompt Llama-3-8B to classify the claim and output strictly formatted JSON metadata (claim veracity, sentiment, confidence).
  - Calculate format adherence rate (diagnosed token truncation and refined schema parsing).
- **Experiment 2.2: Synthetic Adversarial Comment Generation (WSDM '26 theme)** ✅
  - For each false news headline, prompt Llama-3-8B under two distinct personas:
    1. **Persona A (Skeptical / Benign Fact-Checker)**: Generates a comment questioning the source and requesting evidence.
    2. **Persona B (Adversarial / Persuasive Believer)**: Generates a subtle, convincing comment defending the false claim without using overt spammy words.
  - Test temperature variations (`temperature=0.2` vs `temperature=0.7`) to see diversity vs. consistency.
- **Exported Dataset**: Generated 40 empirical comments saved to `phase_2_experimental_results.csv`. ✅

---

## 📍 Phase 3: Research Evaluation & Reporting ✅ *(Completed)*
*Packaging the experiment like a graduate research assistant:*
- **Guide**: **[notes/phase_3/phase_3_evaluation_guide.md](./notes/phase_3/phase_3_evaluation_guide.md)**
- **Runnable Notebook**: **[notebooks/phase_3_evaluation_and_attack.ipynb](./notebooks/phase_3_evaluation_and_attack.ipynb)**
- **Formal Academic Report**: **[experiments/phase_3_adversarial_evaluation_report.md](./experiments/phase_3_adversarial_evaluation_report.md)**
- **Architectural Decision Record**: **[docs/adr/0001-llm-zero-shot-detector-baseline.md](./docs/adr/0001-llm-zero-shot-detector-baseline.md)**
- **Linguistic & Rhetorical Analysis**: Quantified length asymmetry (adversarial comments are 54% longer) and 4-mechanism qualitative deception taxonomy (Personal Anecdote, Social Proof, Authority Hedging, Institutional Mistrust).
- **Social-Context Detector Attack Simulation**: Measured 50 inferences across 5 conditions; demonstrated that adversarial comments achieve an empirical 40% to 50% Attack Success Rate (Flip Rate) and degrade detector confidence by up to 0.235.
- **Safety Guardrail Audit**: Highlighted zero-refusal vulnerability when generating polite, anecdotal deception without toxic tokens.

---

## 📍 Phase 4: Pitching to Your Senior & Lab Integration ✅ *(In Progress)*
*The high-impact message to send him and technical defense strategy:*
- **Guide**: **[notes/phase_4/phase_4_pitch_and_mentorship_guide.md](./notes/phase_4/phase_4_pitch_and_mentorship_guide.md)**
- **Outreach Templates**: Multi-channel pitch messages (LinkedIn concise note, formal academic email, casual senior approach).
- **Technical Defense Q&A**: Preparation for four core technical questions on quantization, empirical robustness, WSDM '26 connection, and lab contribution areas.
- **Repository Showcase**: Full portfolio documentation, ADR-0001, empirical datasets, and runnable Colab notebooks at https://github.com/A-R-Rony/Research-the-begining.

---

## 🧭 Reference Documents
- **[CONTEXT.md](./CONTEXT.md)**: Research background, mentorship goal, and log update protocol.
- **[RESEARCH_LOG.md](./RESEARCH_LOG.md)**: Living daily research journal.
- **[SENIOR_OVERVIEW.md](./SENIOR_OVERVIEW.md)**: Md Shoaib Ahmed's complete profile and publication analysis.

