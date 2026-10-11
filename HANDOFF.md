# Project Handoff: 2-Hour Daily Execution Plan (Phase 1 Onward)

## 1. Executive Summary & Objective
- **Target Mentorship**: Md Shoaib Ahmed (Ph.D. Researcher, Boise State University, Trustworthy AI Lab).
- **Goal**: Run `meta-llama/Meta-Llama-3-8B-Instruct` on Google Colab under 4-bit NF4 quantization on an NVIDIA T4 GPU, then generate synthetic adversarial vs. benign comments on fake news headlines to evaluate detector robustness.
- **Current Status**: Phase 0 is 100% complete. Phase 1 (Operational Milestone: Google Colab T4 Deployment) is actively underway. Notebook and guide generated.

---

## 2. Core Repository Reference Files
- **Phase 1 Implementation (Active)**:
  - `notes/phase_1/phase_1_colab_guide.md` (Step-by-step Colab T4 execution and prerequisites guide).
  - `notebooks/phase_1_llama3_colab.ipynb` (Ready-to-upload Jupyter notebook with all 7 cells).
- **Phase 0 Notes**:
  - `notes/phase_0/topic_0.1_why_8b_parameters.md` (Architecture, 8B parameters, tokens, VRAM math).
  - `notes/phase_0/topic_0.2_bitsandbytes_vram_cheatsheet.md` (4-bit BitsAndBytesConfig, 5.5 GB VRAM footprint, Colab cells).
  - `notes/phase_0/topic_0.3_adversarial_comments_and_detectors.md` (Social detectors, XAI-guided attacks, InstructNet).
- **Master Plan**: `PLAN.md` (Phases 0 through 4 detailed).
- **Daily Journal**: `RESEARCH_LOG.md` (Updated through Day 4, October 11, 2026).
- **Senior Background**: `CONTEXT.md` and `SENIOR_OVERVIEW.md`.

---

## 3. What To Do Next (Day 1: Phase 1 — Google Colab Deployment)
When you start the next session, begin immediately with **Phase 1 (2 hours)**:

### Hour 1: Colab Setup & Authentication
1. Open Google Colab and set **Runtime -> Change runtime type -> T4 GPU**.
2. Run `!nvidia-smi` to verify GPU memory (~15.0 GB VRAM).
3. Install dependencies:
   ```bash
   !pip install -q -U torch transformers bitsandbytes accelerate huggingface_hub
   ```
4. Authenticate Hugging Face with your token (`huggingface_hub.login()`).
5. Ensure gated access to `meta-llama/Meta-Llama-3-8B-Instruct` is accepted.

### Hour 2: Model Loading & First Test
1. Set up the 4-bit config:
   ```python
   from transformers import BitsAndBytesConfig
   import torch

   bnb_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_quant_type="nf4",
       bnb_4bit_use_double_quant=True,
       bnb_4bit_compute_dtype=torch.float16
   )
   ```
2. Load model using `device_map="auto"`.
3. Verify memory usage using `torch.cuda.memory_allocated()` is ~5.5 GB (plenty of free space).
4. Run test generation with `apply_chat_template`.

---

## 4. The Full 5-Day Roadmap (2 Hours / Day)
- **Day 1 (2 hrs)**: Phase 1 — Colab setup, 4-bit Llama-3-8B loading, smoke test.
- **Day 2 (2 hrs)**: Phase 2 Part 1 — LIAR benchmark dataset + JSON instruction following (InstructNet theme).
- **Day 3 (2 hrs)**: Phase 2 Part 2 — Adversarial comment generation under dual personas (WSDM '26 theme).
- **Day 4 (2 hrs)**: Phase 3 — Results evaluation & experiment report (`experiments/adversarial_comment_generation.md`).
- **Day 5 (2 hrs)**: Phase 4 — GitHub cleanup & pitch message to Shoaib bhai.

---

## 5. Instructions for AI Agent in Next Session
- **Formatting Constraint**: ABSOLUTE BAN on `$`, `$$`, `\`, `_{}`, or `^{}`. Always use forward slashes `/` in paths.
- **Interaction Style**: Keep explanations direct and code-focused. Do not get bogged down in deep paper theory. Walk the user step-by-step through Colab cells.
