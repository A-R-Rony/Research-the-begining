# Research Context & Starting Assessment

## 1. My Current Situation & Background
- **Current State**: Beginner in LLM / AI / ML research with no prior formal publications or ML research experience.
- **Goal**: Earn a research mentorship and collaboration spot with a senior researcher (**Md Shoaib Ahmed**, Ph.D. researcher at Boise State University).
- **Senior's Context & Lab**:
  - Focus area: **Trustworthy & Responsible AI, Adversarial LLMs, Misinformation Detection, and Instruction Classification**.
  - Flagship 2025–2026 works: 
    - *XAI-guided Adversarial Comment Generation with LLMs to Attack Fake News Detectors* (ACM WSDM 2026).
    - *Generating Realistic Adversarial User Comment Attacks to Evaluate Fake News Detectors* (IEEE TCSS 2026).
    - *InstructNet: Multi-Label Instruction Classification* (PLOS ONE / arXiv 2024–2025).
- **Senior's Guidance to Me**:
  - Start by learning how to run an **open-weights 8B model (like Llama-3-8B)** in **Google Colab or Kaggle**.
  - Must understand practical hardware limits (VRAM, quantization, tokenization).
  - Prove competence through structured, documented experimentation rather than theoretical talk.

---

## 2. The Core Connection: Why This Matters to His Research
In Shoaib's lab, research requires:
1. **Model Autonomy & Scripting**: Querying open models thousands of times to synthesize adversarial user comments or classify instructions. Proprietary APIs (OpenAI/Anthropic) are cost-prohibitive for large-scale experiments and lack control over logits/seeds.
2. **Resource Constraints**: Running models on university/cloud GPUs requires knowing how to fit 8B parameter models into standard memory budgets (e.g. 16GB VRAM on Colab T4) using **4-bit quantization (bitsandbytes)**.
3. **Prompt Formulation & Evaluation**: Systematically designing prompts (adversarial personas, instruction types) and evaluating the outputs against baseline classifiers.

---

## 3. Protocol for Maintaining and Updating this Research Log
**For any AI Agent working in this repository**:
- When the user says *"update research log"*, *"update today's work"*, or provides notes/reflections on their progress:
  1. Open [RESEARCH_LOG.md](./RESEARCH_LOG.md).
  2. Append a new structured entry (e.g. `## 📅 Day N: [Date] — [Core Focus/Milestone]`) or update the current day's open section.
  3. Include 4 mandatory sections:
     - **What was tackled today** (goals, questions explored).
     - **Concepts understood** (explained in clear, plain language with relevant ML math/terms).
     - **Artifacts / code built** (using repo-relative links e.g. `[FILE](./path)`).
     - **Next day's focus / open questions**.
  4. Keep all links **repo-relative** (do NOT use machine-dependent paths like `file:///C:/...` or `d:/...`).

---

## 4. What I Must Prove to Earn Mentorship
1. **Operational Fluency**: I can spin up an open 8B model in Colab, configure quantization, track VRAM, and generate text via script.
2. **Scientific Curiosity & Discipline**: I don't just "play with ChatGPT"; I formulate a test hypothesis, run comparative prompts (e.g. standard vs. adversarial instruction), record outputs, and calculate evaluation metrics.
3. **Alignment with His Domain**: My practice experiments will center around **instruction following, text classification, and synthetic adversarial comment generation**—the exact problems he publishes on.
