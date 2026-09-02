# Kite — Model Card

## Overview

Kite is an experimental language-model project focused on conversational interaction in Brazilian Portuguese (PT-BR).

## Base model

- **Model:** Qwen2.5-0.5B-Instruct
- **Repository:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Parameters:** approximately 0.49B
- **Planned adaptation:** supervised fine-tuning (SFT) with LoRA

## Current status

The Qwen2.5-0.5B-Instruct base model has been loaded and tested locally with Transformers. Kite fine-tuning has not yet been performed.

## Intended pipeline

```text
Qwen2.5-0.5B-Instruct
        ↓
     SFT + LoRA
        ↓
   Kite adapter
        ↓
      Merge
        ↓
   Kite model
```

## Language

Primary target: Portuguese (Brazil) — `pt-BR`.

The underlying Qwen model is multilingual; Kite's fine-tuning goal is to specialize conversational behavior for Brazilian Portuguese.

## Dataset provenance

Kite training datasets must have documented sources and licenses. Dataset licensing is independent from the base model license.

## License and attribution

See the repository `LICENSE` and the original Qwen model documentation for the applicable license and attribution requirements.

Base model: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
