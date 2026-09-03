"""Treinamento SFT com LoRA para o Kite.

O pipeline suporta:
- datasets single-turn e multivoltas em ``messages``;
- loss somente nos tokens do assistant;
- máscara nativa do chat template quando disponível;
- fallback compatível para templates sem assistant token mask;
- divisão treino/validação para monitorar overfitting;
- dry-run com estatísticas de tokens, loss coverage e truncamento;
- checkpointing por época e carregamento do melhor checkpoint;
- configuração por linha de comando para experimentos reprodutíveis.

Uso:
    python training/train.py --dry-run
    python training/train.py --epochs 2 --yes
    python training/train.py --lr 1e-4 --max-length 1024 --yes
    python training/train.py --resume

A avaliação semântica/factual continua sendo feita separadamente em
``evaluation/prompts.txt``. A validação interna serve para acompanhar o treino,
não para substituir o conjunto de teste externo.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import shutil
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "base_model" / "Qwen2.5-0.5B-Instruct"
DATASET_PATH = ROOT / "datasets" / "processed" / "train.jsonl"
OUTPUT_PATH = ROOT / "outputs" / "kite-lora"

DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION = 8
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_MAX_LENGTH = 1024
DEFAULT_EVAL_RATIO = 0.10
DEFAULT_LOGGING_STEPS = 5
DEFAULT_SEED = 42
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05

ALLOWED_ROLES = {"system", "user", "assistant"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina o Kite com SFT + LoRA usando Transformers e PEFT."
    )
    parser.add_argument("--yes", action="store_true", help="Não pedir confirmação.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Valida, tokeniza e analisa o dataset sem treinar.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Retoma do último checkpoint existente em outputs/kite-lora.",
    )
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=DEFAULT_GRADIENT_ACCUMULATION,
    )
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--eval-ratio", type=float, default=DEFAULT_EVAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Ativa gradient checkpointing para reduzir memória.",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Não cria split de validação. Útil apenas para diagnósticos específicos.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove checkpoints anteriores antes de um novo treino. Não usa com --resume.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs deve ser maior que zero.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size deve ser maior que zero.")
    if args.gradient_accumulation <= 0:
        raise ValueError("--gradient-accumulation deve ser maior que zero.")
    if args.lr <= 0:
        raise ValueError("--lr deve ser maior que zero.")
    if args.max_length < 32:
        raise ValueError("--max-length deve ser pelo menos 32 tokens.")
    if not 0 <= args.eval_ratio < 1:
        raise ValueError("--eval-ratio deve estar entre 0 e 1.")
    if args.lora_r <= 0:
        raise ValueError("--lora-r deve ser maior que zero.")
    if args.lora_alpha <= 0:
        raise ValueError("--lora-alpha deve ser maior que zero.")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora-dropout deve estar entre 0 e 1.")
    if args.clean_output and args.resume:
        raise ValueError("--clean-output e --resume não podem ser usados juntos.")


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


def check_dependencies() -> None:
    missing = []
    for package in ("torch", "transformers", "datasets", "peft", "accelerate"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        raise RuntimeError(
            "Dependências ausentes: " + ", ".join(missing) +
            "\nInstale com:\n"
            "  pip install torch transformers datasets peft accelerate"
        )


def normalize_message(message: Any) -> dict[str, str]:
    if not isinstance(message, dict):
        raise ValueError("cada mensagem deve ser um objeto JSON")

    role = message.get("role")
    content = message.get("content")

    if role not in ALLOWED_ROLES:
        raise ValueError(f"role inválida: {role!r}")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("mensagem com conteúdo vazio ou não textual")

    return {"role": role, "content": content.strip()}


def validate_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("'messages' deve conter pelo menos dois turnos")

    normalized = [normalize_message(message) for message in messages]

    if normalized[0]["role"] != "user":
        raise ValueError("a conversa deve começar com user")
    if normalized[-1]["role"] != "assistant":
        raise ValueError("a conversa deve terminar com assistant")
    if not any(m["role"] == "user" for m in normalized):
        raise ValueError("conversa sem user")
    if not any(m["role"] == "assistant" for m in normalized):
        raise ValueError("conversa sem assistant")

    # Não permitimos dois turnos consecutivos do mesmo papel. Isso evita
    # amostras malformadas e torna a máscara previsível.
    previous_role = None
    for message in normalized:
        role = message["role"]
        if role in {"user", "assistant"} and role == previous_role:
            raise ValueError(f"turnos consecutivos do mesmo papel: {role}")
        if role in {"user", "assistant"}:
            previous_role = role

    return normalized


def load_examples() -> list[dict[str, Any]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset processado não encontrado: {DATASET_PATH}\n\n"
            "Execute primeiro:\n"
            "  python scripts/prepare_dataset.py"
        )

    examples: list[dict[str, Any]] = []
    invalid = 0

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("linha não é um objeto JSON")
                messages = validate_messages(item.get("messages"))
                examples.append({"messages": messages, "line": line_number})
            except (json.JSONDecodeError, ValueError) as exc:
                invalid += 1
                print(f"⚠ Linha {line_number} ignorada: {exc}")

    if not examples:
        raise ValueError("Nenhum exemplo válido foi encontrado no dataset.")

    print(f"✓ Exemplos válidos: {len(examples)}")
    print(f"⚠ Exemplos inválidos: {invalid}")
    return examples


def find_checkpoint() -> str | None:
    if not OUTPUT_PATH.exists():
        return None

    checkpoints = []
    for path in OUTPUT_PATH.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))

    if not checkpoints:
        return None
    return str(max(checkpoints, key=lambda item: item[0])[1])


def maybe_clean_output(args: argparse.Namespace) -> None:
    if not args.clean_output or not OUTPUT_PATH.exists():
        return

    print(f"🧹 Removendo checkpoints anteriores de {OUTPUT_PATH}...")
    for checkpoint in OUTPUT_PATH.glob("checkpoint-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)


def fingerprint_examples(examples: list[dict[str, Any]]) -> str:
    payload = [example["messages"] for example in examples]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def split_examples(examples: list[dict[str, Any]], args: argparse.Namespace):
    from datasets import Dataset

    rows = [{"messages": example["messages"]} for example in examples]
    dataset = Dataset.from_list(rows)

    if args.no_validation or args.eval_ratio == 0 or len(dataset) < 10:
        print("ℹ Validação interna: desativada")
        return dataset, None

    split = dataset.train_test_split(
        test_size=args.eval_ratio,
        seed=args.seed,
        shuffle=True,
    )

    print(f"✓ Split treino: {len(split['train'])}")
    print(f"✓ Split validação: {len(split['test'])}")
    return split["train"], split["test"]


def _as_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        return list(value[0])
    return list(value)


def native_assistant_mask(tokenizer, messages: list[dict[str, str]]) -> tuple[list[int], list[int]] | None:
    """Usa a máscara oficial do chat template quando o template oferece suporte."""
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
    except (TypeError, ValueError, KeyError, NotImplementedError):
        return None

    if not isinstance(encoded, dict):
        return None

    ids = encoded.get("input_ids")
    mask = encoded.get("assistant_masks")
    if mask is None:
        mask = encoded.get("assistant_tokens_mask")
    if ids is None or mask is None:
        return None

    input_ids = _as_list(ids)
    assistant_mask = _as_list(mask)
    if len(input_ids) != len(assistant_mask):
        return None

    return input_ids, assistant_mask


def fallback_assistant_mask(
    tokenizer,
    messages: list[dict[str, str]],
) -> tuple[list[int], list[int]]:
    """Fallback para chat templates sem {% generation %}."""
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    mask = [0] * len(full_ids)

    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        before = tokenizer.apply_chat_template(
            messages[:index],
            tokenize=False,
            add_generation_prompt=True,
        )
        through = tokenizer.apply_chat_template(
            messages[: index + 1],
            tokenize=False,
            add_generation_prompt=False,
        )

        start = len(tokenizer(before, add_special_tokens=False)["input_ids"])
        end = len(tokenizer(through, add_special_tokens=False)["input_ids"])
        start = max(0, min(start, len(full_ids)))
        end = max(start, min(end, len(full_ids)))
        for position in range(start, end):
            mask[position] = 1

    return list(full_ids), mask


def encode_messages(
    tokenizer,
    messages: list[dict[str, str]],
) -> tuple[list[int], list[int], str]:
    native = native_assistant_mask(tokenizer, messages)
    if native is not None:
        ids, mask = native
        return ids, mask, "native"

    ids, mask = fallback_assistant_mask(tokenizer, messages)
    return ids, mask, "fallback"


def trim_sequence(
    input_ids: list[int],
    assistant_mask: list[int],
    max_length: int,
) -> tuple[list[int], list[int], bool]:
    if len(input_ids) <= max_length:
        return input_ids, assistant_mask, False

    # Mantemos o final da conversa, porque em exemplos multivoltas o turno mais
    # recente costuma depender do contexto imediatamente anterior. Ajustamos a
    # máscara junto com a janela.
    start = len(input_ids) - max_length
    return input_ids[start:], assistant_mask[start:], True


def build_tokenized_dataset(tokenizer, dataset, max_length: int):
    from datasets import Dataset

    rows = []
    skipped = 0
    truncated = 0
    native_mask_count = 0
    fallback_mask_count = 0
    total_input_tokens = 0
    total_assistant_tokens = 0
    multiturn = 0

    for example in dataset:
        messages = example["messages"]
        try:
            input_ids, assistant_mask, mask_mode = encode_messages(tokenizer, messages)
            if mask_mode == "native":
                native_mask_count += 1
            else:
                fallback_mask_count += 1

            input_ids, assistant_mask, was_truncated = trim_sequence(
                input_ids,
                assistant_mask,
                max_length,
            )

            if was_truncated:
                truncated += 1

            labels = [
                token if active else -100
                for token, active in zip(input_ids, assistant_mask)
            ]

            active = sum(label != -100 for label in labels)
            if len(input_ids) < 2 or active == 0:
                skipped += 1
                continue

            rows.append({
                "input_ids": input_ids,
                "labels": labels,
            })
            total_input_tokens += len(input_ids)
            total_assistant_tokens += active
            if len(messages) > 2:
                multiturn += 1
        except Exception as exc:
            skipped += 1
            print(f"⚠ Exemplo ignorado durante tokenização: {exc}")

    if not rows:
        raise ValueError("Nenhum exemplo pôde ser tokenizado.")

    print(f"✓ Exemplos tokenizados: {len(rows)}")
    print(f"✓ Multivoltas: {multiturn}")
    print(f"✓ Truncados: {truncated}")
    print(f"✓ Máscara nativa do template: {native_mask_count}")
    print(f"✓ Máscara fallback: {fallback_mask_count}")
    print(f"✓ Tokens de entrada: {total_input_tokens}")
    print(f"✓ Tokens com loss: {total_assistant_tokens}")
    print(f"✓ Cobertura de loss: {total_assistant_tokens / max(total_input_tokens, 1):.2%}")
    if skipped:
        print(f"⚠ Ignorados após tokenização: {skipped}")

    return Dataset.from_list(rows)


def dataset_stats(dataset) -> dict[str, Any]:
    lengths = [len(row["input_ids"]) for row in dataset]
    assistant_tokens = [
        sum(label != -100 for label in row["labels"])
        for row in dataset
    ]

    return {
        "examples": len(dataset),
        "input_tokens": sum(lengths),
        "assistant_tokens": sum(assistant_tokens),
        "loss_coverage": round(sum(assistant_tokens) / max(sum(lengths), 1), 6),
        "mean_input_tokens": round(sum(lengths) / max(len(lengths), 1), 2),
        "mean_assistant_tokens": round(
            sum(assistant_tokens) / max(len(assistant_tokens), 1),
            2,
        ),
        "max_input_tokens": max(lengths, default=0),
        "min_input_tokens": min(lengths, default=0),
    }


def save_run_manifest(
    args: argparse.Namespace,
    raw_fingerprint: str,
    train_stats: dict[str, Any],
    eval_stats: dict[str, Any] | None,
    mask_mode: str,
) -> None:
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": str(MODEL_PATH),
        "dataset": str(DATASET_PATH),
        "dataset_sha256": raw_fingerprint,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.lr,
            "max_length": args.max_length,
            "seed": args.seed,
            "eval_ratio": args.eval_ratio if not args.no_validation else 0,
            "mask_mode": mask_mode,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        },
        "train_stats": train_stats,
        "eval_stats": eval_stats,
    }
    (OUTPUT_PATH / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def make_training_args(TrainingArguments, args, torch):
    params = {
        "output_dir": str(OUTPUT_PATH),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": DEFAULT_LOGGING_STEPS,
        "logging_first_step": True,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "fp16": torch.cuda.is_available(),
        "bf16": False,
        "report_to": "none",
        "run_name": "Kite-LoRA",
        "seed": args.seed,
        "data_seed": args.seed,
        "remove_unused_columns": False,
        "dataloader_pin_memory": torch.cuda.is_available(),
        "optim": "adamw_torch",
        "save_safetensors": True,
        "label_names": ["labels"],
    }

    if args.gradient_checkpointing:
        params["gradient_checkpointing"] = True

    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in signature:
        params["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in signature:
        params["evaluation_strategy"] = "epoch"

    if "load_best_model_at_end" in signature:
        params["load_best_model_at_end"] = True
        params["metric_for_best_model"] = "eval_loss"
        params["greater_is_better"] = False

    # Mantém compatibilidade com versões que ainda não possuem algumas opções.
    supported = {key: value for key, value in params.items() if key in signature}
    return TrainingArguments(**supported)


def make_trainer(Trainer, tokenizer, model, training_args, train_dataset, eval_dataset, collator):
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
    }
    signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in signature:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature:
        kwargs["tokenizer"] = tokenizer
    return Trainer(**kwargs)


def print_banner(args: argparse.Namespace) -> None:
    print("\n╔══════════════════════════════════════════╗")
    print("║             🪁 KITE TRAINING             ║")
    print("║       Qwen2.5-0.5B + LoRA / SFT         ║")
    print("╚══════════════════════════════════════════╝\n")
    print(f"Modelo:  {MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Saída:   {OUTPUT_PATH}")
    print(f"Épocas:  {args.epochs}")
    print(f"Batch:   {args.batch_size} x {args.gradient_accumulation}")
    print(f"LR:      {args.lr}")
    print(f"Contexto máximo: {args.max_length} tokens")
    print(f"Validação: {0 if args.no_validation else args.eval_ratio:.0%}")
    print(
        f"LoRA:    r={args.lora_r}, alpha={args.lora_alpha}, "
        f"dropout={args.lora_dropout}"
    )
    print("Loss:    somente tokens do assistant\n")


def dry_run(args: argparse.Namespace, examples: list[dict[str, Any]]) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "right"

    raw_fingerprint = fingerprint_examples(examples)
    train_raw, eval_raw = split_examples(examples, args)
    train_dataset = build_tokenized_dataset(tokenizer, train_raw, args.max_length)
    eval_dataset = (
        build_tokenized_dataset(tokenizer, eval_raw, args.max_length)
        if eval_raw is not None
        else None
    )

    train_stats = dataset_stats(train_dataset)
    eval_stats = dataset_stats(eval_dataset) if eval_dataset is not None else None

    print("\n✓ DRY-RUN concluído")
    print(f"  Treino: {train_stats['examples']} exemplos")
    if eval_stats:
        print(f"  Validação: {eval_stats['examples']} exemplos")
    print(f"  Loss coverage (treino): {train_stats['loss_coverage']:.2%}")
    print(f"  Média de tokens de entrada: {train_stats['mean_input_tokens']}")
    print(f"  Média de tokens com loss: {train_stats['mean_assistant_tokens']}")
    print(f"  Fingerprint do dataset: {raw_fingerprint[:16]}...")
    print("  Nenhum peso foi alterado.")


def train(args: argparse.Namespace, examples: list[dict[str, Any]]) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    maybe_clean_output(args)

    if not args.yes:
        answer = input("\n⚠ Iniciar treinamento agora? [s/N]: ").strip().lower()
        if answer not in {"s", "sim", "y", "yes"}:
            print("Treinamento cancelado.")
            return

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_name = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"\nDispositivo: {device_name}")
    print(f"Dtype: {dtype}")

    train_raw, eval_raw = split_examples(examples, args)
    print("\n🔤 Tokenizando treino...")
    train_dataset = build_tokenized_dataset(tokenizer, train_raw, args.max_length)
    eval_dataset = None
    if eval_raw is not None:
        print("\n🔤 Tokenizando validação...")
        eval_dataset = build_tokenized_dataset(tokenizer, eval_raw, args.max_length)

    train_stats = dataset_stats(train_dataset)
    eval_stats = dataset_stats(eval_dataset) if eval_dataset is not None else None
    raw_fingerprint = fingerprint_examples(examples)

    # Usa a máscara nativa quando disponível. Se o template não suportar a
    # funcionalidade, o builder registra a quantidade de fallbacks no log.
    mask_mode = "native_or_fallback"
    save_run_manifest(args, raw_fingerprint, train_stats, eval_stats, mask_mode)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
        return_tensors="pt",
    )

    training_args = make_training_args(TrainingArguments, args, torch)

    trainer = make_trainer(
        Trainer,
        tokenizer,
        model,
        training_args,
        train_dataset,
        eval_dataset,
        collator,
    )

    resume_checkpoint = find_checkpoint() if args.resume else None
    if args.resume:
        if resume_checkpoint:
            print(f"\n↩ Retomando de: {resume_checkpoint}")
        else:
            print("\n⚠ --resume foi informado, mas nenhum checkpoint foi encontrado.")
            print("  Um treinamento novo será iniciado.")

    print("\n📊 Resumo do sinal de treinamento")
    print(f"  Exemplos de treino: {train_stats['examples']}")
    print(f"  Tokens de entrada: {train_stats['input_tokens']}")
    print(f"  Tokens do assistant com loss: {train_stats['assistant_tokens']}")
    print(f"  Cobertura de loss: {train_stats['loss_coverage']:.2%}")
    if eval_stats:
        print(f"  Exemplos de validação: {eval_stats['examples']}")

    print("\n🚀 Iniciando treinamento...\n")
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    print("\n💾 Salvando melhor estado do adapter...")
    trainer.save_model(str(OUTPUT_PATH))
    tokenizer.save_pretrained(str(OUTPUT_PATH))

    model.config.use_cache = True
    model.save_pretrained(str(OUTPUT_PATH))

    print("\n✓ Treinamento concluído.")
    print(f"✓ Adapter salvo em: {OUTPUT_PATH}")
    print("✓ Modelo-base não foi sobrescrito.")
    print("✓ training_manifest.json salvo para reprodutibilidade.")
    print("ℹ Para produzir um modelo standalone, use scripts/merge_lora.py.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    check_dependencies()
    set_seed(args.seed)
    print_banner(args)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo-base não encontrado: {MODEL_PATH}\n"
            "Coloque Qwen2.5-0.5B-Instruct em base_model/."
        )

    examples = load_examples()

    if args.dry_run:
        dry_run(args, examples)
        return

    train(args, examples)


if __name__ == "__main__":
    main()
