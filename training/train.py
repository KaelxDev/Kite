"""Treinamento SFT com LoRA para o Kite.

Uso:
    python training/train.py
    python training/train.py --yes
    python training/train.py --dry-run

O pipeline usa Transformers + PEFT + Datasets diretamente. O dataset pode
conter conversas de turno único ou multivoltas em formato `messages`.
A perda é calculada somente sobre tokens associados a mensagens `assistant`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "base_model" / "Qwen2.5-0.5B-Instruct"
DATASET_PATH = ROOT / "datasets" / "processed" / "train.jsonl"
OUTPUT_PATH = ROOT / "outputs" / "kite-lora"

EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 2e-4
MAX_LENGTH = 1024
LOGGING_STEPS = 1
SAVE_STEPS = 50
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o Kite com LoRA")
    parser.add_argument("--yes", action="store_true", help="Não pedir confirmação")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida modelo/dataset e tokeniza amostras sem treinar",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma do último checkpoint disponível na saída",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def check_dependencies():
    missing = []
    for package in ("torch", "transformers", "datasets", "peft"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        raise RuntimeError(
            "Dependências ausentes: " + ", ".join(missing) +
            "\nInstale com:\n"
            "  pip install torch transformers datasets peft accelerate pyyaml"
        )


def load_examples() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset processado não encontrado: {DATASET_PATH}\n\n"
            "Execute primeiro:\n"
            "  python scripts/prepare_dataset.py\n\n"
            "O arquivo esperado é:\n"
            "  datasets/processed/train.jsonl"
        )

    examples = []
    invalid = 0

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid += 1
                print(f"⚠ JSON inválido na linha {line_number}: {exc}")
                continue

            messages = item.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                invalid += 1
                continue

            valid_messages = all(
                isinstance(message, dict)
                and message.get("role") in {"system", "user", "assistant"}
                and isinstance(message.get("content"), str)
                and message["content"].strip()
                for message in messages
            )
            if not valid_messages:
                invalid += 1
                continue

            if not any(message["role"] == "user" for message in messages):
                invalid += 1
                continue
            if not any(message["role"] == "assistant" for message in messages):
                invalid += 1
                continue

            examples.append({"messages": messages})

    if not examples:
        raise ValueError("Nenhum exemplo válido foi encontrado no dataset.")

    print(f"✓ Exemplos válidos: {len(examples)}")
    if invalid:
        print(f"⚠ Exemplos ignorados: {invalid}")

    return examples


def tokenized_assistant_spans(tokenizer, messages: list[dict]) -> tuple[list[int], list[tuple[int, int]]]:
    """Renderiza uma conversa e retorna os intervalos de tokens dos turnos assistant."""
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    spans: list[tuple[int, int]] = []

    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        before_text = tokenizer.apply_chat_template(
            messages[:index],
            tokenize=False,
            add_generation_prompt=True,
        )
        through_text = tokenizer.apply_chat_template(
            messages[: index + 1],
            tokenize=False,
            add_generation_prompt=False,
        )

        start = len(tokenizer(before_text, add_special_tokens=False)["input_ids"])
        end = len(tokenizer(through_text, add_special_tokens=False)["input_ids"])

        if end > start:
            spans.append((start, min(end, len(full_ids))))

    return full_ids, spans


def build_dataset(tokenizer, examples: list[dict]):
    from datasets import Dataset

    rows = []
    skipped = 0
    multiturn = 0

    for example in examples:
        messages = example["messages"]

        try:
            full_ids, assistant_spans = tokenized_assistant_spans(tokenizer, messages)

            if len(messages) > 2:
                multiturn += 1

            if len(full_ids) < 2 or not assistant_spans:
                skipped += 1
                continue

            # Truncamos somente aqui, depois de encontrar os spans na sequência
            # completa, para não perder a posição dos turnos anteriores.
            input_ids = full_ids[:MAX_LENGTH]
            labels = [-100] * len(input_ids)

            for start, end in assistant_spans:
                start = min(start, MAX_LENGTH)
                end = min(end, MAX_LENGTH)
                if start < end:
                    labels[start:end] = input_ids[start:end]

            if not any(label != -100 for label in labels):
                skipped += 1
                continue

            rows.append({"input_ids": input_ids, "labels": labels})
        except Exception as exc:
            skipped += 1
            print(f"⚠ Exemplo ignorado durante tokenização: {exc}")

    if not rows:
        raise ValueError("Nenhum exemplo pôde ser tokenizado.")

    print(f"✓ Exemplos tokenizados: {len(rows)}")
    print(f"✓ Conversas multivoltas detectadas: {multiturn}")
    if skipped:
        print(f"⚠ Exemplos ignorados na tokenização: {skipped}")

    return Dataset.from_list(rows)


def train(args: argparse.Namespace) -> None:
    check_dependencies()
    set_seed(SEED)

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, TaskType, get_peft_model

    print("\n╔══════════════════════════════════════════╗")
    print("║             🪁 KITE TRAINING             ║")
    print("║       Qwen2.5-0.5B + LoRA / SFT         ║")
    print("╚══════════════════════════════════════════╝\n")
    print(f"Modelo:  {MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Saída:   {OUTPUT_PATH}")
    print(f"Épocas:  {EPOCHS}")
    print(f"Batch:   {BATCH_SIZE} x {GRADIENT_ACCUMULATION}")
    print(f"LR:      {LEARNING_RATE}")
    print(f"Contexto máximo: {MAX_LENGTH} tokens")
    print(f"LoRA:    r={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    print("Loss:    somente turnos assistant\n")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo-base não encontrado: {MODEL_PATH}\n"
            "Coloque Qwen2.5-0.5B-Instruct em base_model/."
        )

    examples = load_examples()

    if args.dry_run:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        dataset = build_dataset(tokenizer, examples)
        sample = dataset[0]
        active_labels = sum(label != -100 for label in sample["labels"])
        print("\n✓ DRY-RUN concluído")
        print(f"  Exemplos: {len(dataset)}")
        print(f"  Tokens da primeira amostra: {len(sample['input_ids'])}")
        print(f"  Tokens com loss na primeira amostra: {active_labels}")
        print("  Nenhum peso foi alterado.")
        return

    if not args.yes:
        answer = input("\n⚠ Iniciar treinamento agora? [s/N]: ").strip().lower()
        if answer not in {"s", "sim", "y", "yes"}:
            print("Treinamento cancelado.")
            return

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"\nDispositivo: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = build_dataset(tokenizer, examples)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_PATH),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_strategy="steps",
        save_total_limit=2,
        fp16=use_fp16,
        bf16=False,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print("\n🚀 Iniciando treinamento...\n")
    trainer.train(resume_from_checkpoint=args.resume)

    print("\n💾 Salvando adapter LoRA...")
    trainer.save_model(str(OUTPUT_PATH))
    tokenizer.save_pretrained(str(OUTPUT_PATH))

    print("\n✓ Treinamento concluído.")
    print(f"✓ Adapter salvo em: {OUTPUT_PATH}")
    print("ℹ O modelo-base não foi sobrescrito.")
    print("ℹ Para produzir um modelo standalone, use scripts/merge_lora.py.")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
