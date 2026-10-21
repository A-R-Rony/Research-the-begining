# 07: Structured Epistemic Chain-of-Thought and Self-Consistency Engine

**What to build:** An advanced reasoning module and conceptual guide analyzing why unconstrained 8B models suffer from narrative drift, implementing a three-stage epistemic CoT schema emitted in structured JSON before the verdict, and evaluating Self-Consistency sampling across multiple reasoning paths.

**Blocked by:** 03 (Zero-Shot Veracity Baseline and Failure Boundary Analysis), 05 (Context-Aware Evidence Grounding and Faithfulness Verification)

**Status:** done

- [x] Write conceptual guide on Chain-of-Thought reasoning mechanics, autoregressive token conditioning, and 8B drift failure modes.
- [x] Implement a three-stage epistemic CoT schema enforcing `source_and_entity_check`, `evidence_triangulation`, and `fallacy_or_distortion_detection` prior to the terminal label in JSON.
- [x] Measure token generation latency and output token overhead of CoT compared to cold Zero-Shot inference.
- [x] Implement a Self-Consistency sampling demonstration (k = 3 reasoning trajectories at temperature = 0.7) and majority voting aggregation.

