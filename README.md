# Research-the-begining: Applied LLM Learning & Research Journey

An applied Large Language Model (LLM) research learning roadmap designed to bridge foundational understanding to empirical research. 

This repository directly prepares for collaborative research with **Md Shoaib Ahmed** (Ph.D. Researcher at Boise State University) in **Trustworthy AI, Adversarial LLM generation, and Misinformation Detection**.

---

## 🧭 Project Roadmap & Documents

1. **[RESEARCH_LOG.md](./RESEARCH_LOG.md)**:
   - Living daily research journal tracking concepts learned, experiments run, and daily progress.
2. **[CONTEXT.md](./CONTEXT.md)**:
   - Detailed current assessment, mentorship goals, and connection to the senior's research lab.
3. **[PLAN.md](./PLAN.md)**:
   - Phase-by-phase learning plan from conceptual basics -> Colab 8B model execution -> adversarial comment generation experiments.
4. **[SENIOR_OVERVIEW.md](./SENIOR_OVERVIEW.md)**:
   - Comprehensive profile of Md Shoaib Ahmed, citation metrics (767+ citations, h-index: 12), and complete publication catalog (2023–2026).
5. **Phase 1 Deployment**:
   - **[notes/phase_1/phase_1_colab_guide.md](./notes/phase_1/phase_1_colab_guide.md)**: Step-by-step Colab T4 deployment and VRAM guide.
   - **[notebooks/phase_1_llama3_colab.ipynb](./notebooks/phase_1_llama3_colab.ipynb)**: Executable Google Colab notebook for 4-bit Llama-3-8B.
6. **Phase 2 Research Experiment**:
   - **[notes/phase_2/phase_2_experiment_guide.md](./notes/phase_2/phase_2_experiment_guide.md)**: Research guide for dual-persona adversarial comment generation and JSON instruction adherence.
   - **[notebooks/phase_2_adversarial_comments.ipynb](./notebooks/phase_2_adversarial_comments.ipynb)**: Executable Google Colab research experiment notebook.
7. **Domain Knowledge & Tooling Guides**:
   - **[GLOSSARY.md](./GLOSSARY.md)**: Canonical definitions for LLM research, runtime state, and tensor concepts.
   - **[notes/tooling/README.md](./notes/tooling/README.md)**: Deep dive tooling stack guide (NumPy, PyTorch, Transformers, Pandas) tailored for C# and TypeScript engineers.

---

## 🔬 Core Applied Focus: Open 8B Models (Llama-3-8B) on Constrained Hardware
- **Model Target**: `meta-llama/Meta-Llama-3-8B-Instruct`
- **Hardware Target**: Google Colab (Free NVIDIA T4 GPU / 16GB VRAM)
- **Methodology**: 4-bit NormalFloat Quantization (`bitsandbytes`), prompt engineering for instruction following, and adversarial synthetic comment generation.

