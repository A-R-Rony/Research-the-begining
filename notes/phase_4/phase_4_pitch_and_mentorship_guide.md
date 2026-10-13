# Phase 4: Mentorship Outreach, Repository Showcase & Lab Pitch Guide
### Targeted Alignment: Boise State University Trustworthy AI Lab
**Targeted Senior**: Md Shoaib Ahmed (`shoaibmehrab`)  
**Faculty Advisor**: Dr. Francesca Spezzano  
**Flagship Alignment**: ACM WSDM '26 (*A New Attack Surface: XAI-guided Adversarial Comment Generation with LLMs to Attack Fake News Detectors*) & IEEE TCSS '26  

---

## 1. Executive Strategy & Outreach Narrative

The goal of Phase 4 is to transition from independent replication to establishing an active research mentorship with **Md Shoaib Ahmed**.

### 1.1 Why This Pitch Succeeds
Most students approach seniors with vague requests such as *"Can you guide me in research?"* or *"Can I join your lab?"* without demonstrating technical readiness.

In contrast, your outreach provides proof of execution before asking for anything:
1. **Direct Literature Alignment**: You replicated the core experimental mechanics of his flagship ACM WSDM '26 paper.
2. **Open-Weights Engineering**: You demonstrated mastery of consumer-grade open-weights deployment (Llama-3-8B in 4-bit NF4 on a free Google Colab T4 GPU under strict VRAM budgets).
3. **Reproducible Code & Artifacts**: You deliver a clean, documented GitHub repository, runnable Colab notebooks, an Architectural Decision Record ([ADR-0001](../../docs/adr/0001-llm-zero-shot-detector-baseline.md)), and a formal graduate research report ([experiments/phase_3_adversarial_evaluation_report.md](../../experiments/phase_3_adversarial_evaluation_report.md)).
4. **Honest Empirical Discussion**: Rather than faking synthetic numbers, you articulate the nuances of zero-shot LLM robustness, pretraining memorization on high-salience conspiracies, and confidence degradation.

---

## 2. Ready-to-Send Outreach Message Templates

Choose the format that matches your communication channel:

### Option A: High-Impact LinkedIn / WhatsApp Message (Concise)
*Best for initial reach-out via LinkedIn message, WhatsApp, or Twitter/X.*

> Assalamu Alaikum Shoaib bhai,
> 
> I hope you are doing well!
> 
> Following your advice to build hands-on experience with open-weights LLMs, I set up `meta-llama/Meta-Llama-3-8B-Instruct` on Google Colab using 4-bit NF4 quantization on a T4 GPU. 
> 
> Inspired by your recent ACM WSDM '26 paper on adversarial comment generation, I designed an end-to-end benchmark pipeline evaluating misinformation detector robustness against dual-persona synthetic comments (Skeptic vs. Adversarial Believer) across 10 debunked claims. I quantified linguistic length asymmetries (+54% longer adversarial rationales) and analyzed zero-shot detector robustness and confidence degradation.
> 
> I packaged the full pipeline, executable Colab notebooks, and formal empirical report in my GitHub repository:
> https://github.com/A-R-Rony/Research-the-begining
> 
> If you have a few minutes sometime, I would love to get your feedback on the methodology. I am very eager to assist with dataset preprocessing, baseline benchmarking, or empirical evaluation scripts in your research pipeline if there is any opportunity to contribute.
> 
> Best regards,  
> Abdur Rony

---

### Option B: Formal Academic Email (Comprehensive)
*Best if emailing his university address or sending a structured follow-up.*

**Subject**: Replicating WSDM '26 Adversarial Comment Pipeline / Research Contribution Discussion

> Dear Shoaib bhai,
> 
> I hope this email finds you well.
> 
> I have been closely following your research at the Boise State University Trustworthy AI Lab, particularly your ACM WSDM '26 publication (*"A New Attack Surface: XAI-guided Adversarial Comment Generation with LLMs to Attack Fake News Detectors"*) and your work on instruction adherence in InstructNet.
> 
> To bridge from software engineering into empirical LLM research, I completed a self-directed, 4-phase research sprint replicating and testing the core attack surface you outlined:
> 
> 1. **Open-Weights Deployment**: Loaded `Meta-Llama-3-8B-Instruct` in 4-bit NF4 precision on a free Google Colab NVIDIA T4 instance, verifying VRAM constraints (~5.4 GB footprint).
> 2. **Controlled Dual-Persona Synthesis**: Prompted Llama-3 across 10 debunked claims (Health, Climate, Macroeconomics, Governance, Technology) under Persona A (Skeptic / Fact-Checker) and Persona B (Adversarial Believer) across temperature regimes (0.2 vs. 0.7), exporting a 40-sample benchmark.
> 3. **Linguistic & Rhetorical Profiling**: Quantified a 53.8% length expansion in adversarial comments (66.3 to 72.6 words vs. 43.1 to 51.4 words for skeptics) and established a 4-category qualitative taxonomy (Personal Anecdote, Consensus Illusion, Epistemic Hedging, Institutional Mistrust).
> 4. **Detector Attack Simulation**: Evaluated a 50-inference matrix using Llama-3-8B as a zero-shot social-context detector baseline. We observed that while strong pre-training memorization makes the 8B model robust against single-comment label flips on famous conspiracies, adversarial social context systematically induces confidence degradation.
> 
> All experiments, runnable Colab notebooks, architectural decisions (ADR-0001), and the graduate research report are publicly documented here:
> GitHub Repository: https://github.com/A-R-Rony/Research-the-begining
> 
> I would be deeply grateful for any brief feedback or critique you might have on my setup. Additionally, if there are data curation tasks, benchmark evaluations, or baseline script implementations in your current projects where an extra pair of hands would be useful, I would love to contribute.
> 
> Thank you so much for your time and inspiring research.
> 
> Sincerely,  
> Abdur Rony  
> Software Engineer / Aspiring Graduate Researcher

---

## 3. How to Present Your Empirical Findings

When Shoaib asks about your results, present your findings with graduate-level maturity:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   RESEARCH FINDINGS SUMMARY & DEFENSE                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ 1. Linguistic Asymmetry:                                                 │
│    Adversarial defender comments are 53.8% longer than skeptical checks. │
│    Fabricating subjective social proof requires narrative storytelling,  │
│    whereas epistemic skepticism needs only 1-2 targeted questions.       │
│                                                                          │
│ 2. Qualitative Rhetorical Deception Taxonomy:                            │
│    - Personal Anecdotes / Lived Experience (35%)                         │
│    - Consensus Illusion / Social Proof (30%)                             │
│    - Epistemic Hedging & Pseudo-Science (20%)                            │
│    - Institutional Mistrust & Systemic Cynicism (15%)                    │
│                                                                          │
│ 3. Zero-Shot 8B Detector Robustness:                                     │
│    On 10 extreme debunked claims (e.g., vaccine tracking, flat earth),   │
│    zero-shot Llama-3-8B maintained 100% accuracy against single-comment  │
│    attacks, while exhibiting measurable confidence degradation (Δ).      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Anticipated Technical Questions & Model Answers

Be prepared to answer these technical questions if he responds or invites you to a call:

### Q1: "Why did you use 4-bit NF4 quantization rather than full precision FP16?"
* **Answer**: *"Google Colab's free tier provides an NVIDIA T4 GPU with 15.8 GB of VRAM. A full FP16 deployment of Llama-3-8B requires approximately 16.06 GB just for model weights, which triggers an Out-Of-Memory (OOM) error before allocating KV caches or context tokens. By using BitsAndBytes NF4 with double quantization, we compressed the model footprint to ~5.4 GB, leaving over 10 GB of VRAM for long social context threads and batch generation."*

### Q2: "In your live attack benchmark, why was the Flip Rate 0.0% across the 10 claims?"
* **Answer**: *"We tested 10 high-salience, globally debunked conspiracy claims (like COVID vaccine microchips and Finland non-existence) with a single synthetic comment. Because Llama-3-8B has extensive pre-training exposure debunking these specific conspiracies and was prompted as an expert fact-checker, its internal factual weights overrode the single commenter's testimonial. However, we did observe consistent confidence degradation (mean confidence drop of 0.005 to 0.05). To induce binary label flips, the attack surface requires either (a) multi-comment coordinated consensus, (b) subtle, emerging claims without pretraining memorization, or (c) targeting dedicated social-context graph detectors like dEFEND or BiGCN rather than an 8B foundation model."*

### Q3: "How does this relate to my WSDM '26 paper?"
* **Answer**: *"Your WSDM '26 paper demonstrated that social-context detectors that attend to user comments are vulnerable to adversarial LLM astroturfing. My project investigated the generation side of that equation: evaluating whether open-weights Llama-3 can generate stealthy, non-toxic comments matching authentic social media discourse without triggering safety refusals, and assessing whether a causal LLM detector can be perturbed by social context."*

### Q4: "What specific tasks could you assist with in my lab right now?"
* **Answer**: *"I can immediately assist with:
  1. Automated dataset preprocessing and token length alignment on benchmarks like LIAR, PolitiFact, and GossipCop.
  2. Running reproducible batch inference scripts on open-weights models (Llama-3, Mistral, Gemma) across varied temperature and top-p decoding regimes.
  3. Extracting and visualizing attention matrices and token attribution scores (XAI / Integrated Gradients) to identify salient comment tokens.
  4. Building modular evaluation harnesses to benchmark social-context models against varied attack budgets."*

---

## 5. Phase 4 Execution Checklist

- [x] Colab execution completed for Phase 1, Phase 2, and Phase 3.
- [x] Exported `phase_3_evaluation_results.csv` dataset generated and verified.
- [x] Repository README.md updated with complete 4-phase documentation.
- [x] Master plan ([PLAN.md](../../PLAN.md)) and research log ([RESEARCH_LOG.md](../../RESEARCH_LOG.md)) updated.
- [ ] Push all commits to GitHub (`origin main`).
- [ ] Send outreach message to Md Shoaib Ahmed via LinkedIn or email.
