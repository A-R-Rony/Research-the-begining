# 01: Open-Weight Prompt Serialization and KV Cache Memory Guide

**What to build:** An educational and mechanistic guide that demystifies how instruction-tuned open-weight language models tokenize text, compile chat templates with special delimiters, and consume GPU VRAM as context expands, providing verifiable mathematical formulas and code examples.

**Blocked by:** None (can start immediately)

**Status:** done

- [x] Explain byte-pair encoding (BPE) tokenization mechanics and the 128k vocabulary lookup table in Llama-3.
- [x] Detail the exact special token schema (`<|begin_of_text|>`, `<|start_header_id|>system<|end_header_id|>`, `<|eot_id|>`).
- [x] Implement and verify `tokenizer.apply_chat_template` compiling multi-turn role dictionaries into raw prompt tensors.
- [x] Provide mathematical formulas for KV cache memory footprint per token and context window budgeting on a 15 GB Colab T4 GPU.

