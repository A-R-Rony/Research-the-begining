# 04: Few-Shot In-Context Learning: Native Turns vs Monolithic Packing

**What to build:** An in-depth pedagogical guide and comparative experimental harness demonstrating how in-context demonstration exemplars guide model activation space without parameter updates, testing native multi-turn chat templates against monolithic prompt packing across k = 1, 3, and 5 exemplars.

**Blocked by:** 01 (Open-Weight Prompt Serialization and KV Cache Memory Guide), 02 (Curate 20-Claim Benchmark Suite and Knowledge Corpus), 03 (Zero-Shot Veracity Baseline and Failure Boundary Analysis)

**Status:** done

- [x] Write conceptual guide explaining In-Context Learning (ICL) activation steering and attention routing versus gradient fine-tuning.
- [x] Implement native multi-turn few-shot formatting where each exemplar is compiled via `apply_chat_template` as alternating user/assistant turns.
- [x] Implement monolithic in-prompt few-shot formatting where all demonstrations are serialized within a single user turn string.
- [x] Evaluate both formatting styles across k = 1, 3, and 5 exemplars, recording schema adherence, format drift, and token overhead.

