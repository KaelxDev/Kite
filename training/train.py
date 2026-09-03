"""Treinamento SFT + LoRA do Kite.

A configuração experimental fica centralizada em ``training/config.yaml``.
Os argumentos de CLI são apenas overrides temporários e não definem defaults
independentes do arquivo de configuração.
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
CONFIG_PATH = ROOT / "training" / "config.yaml"
ALLOWED_ROLES = {"system", "user", "assistant"}


def load_config() -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML não está instalado. Instale com: pip install pyyaml") from exc
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuração não encontrada: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError("training/config.yaml deve conter um objeto YAML.")
    return config


def get_section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Seção inválida no config.yaml: {name}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o Kite com SFT + LoRA usando training/config.yaml.")
    parser.add_argument("--config", default=str(CONFIG_PATH), help=argparse.SUPPRESS)
    parser.add_argument("--yes", action="store_true", help="Não pedir confirmação.")
    parser.add_argument("--dry-run", action="store_true", help="Valida e tokeniza sem treinar.")
    parser.add_argument("--resume", action="store_true", help="Retoma do último checkpoint.")
    parser.add_argument("--epochs", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--eval-ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lora-r", type=int)
    parser.add_argument("--lora-alpha", type=int)
    parser.add_argument("--lora-dropout", type=float)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def build_settings(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    training = get_section(config, "training")
    lora = get_section(config, "lora")
    experiment = get_section(config, "experiment")
    settings = {
        "model_name_or_path": config.get("model_name_or_path"),
        "dataset_path": config.get("dataset_path"),
        "output_dir": config.get("output_dir"),
        "epochs": training.get("num_train_epochs"),
        "batch_size": training.get("per_device_train_batch_size"),
        "gradient_accumulation": training.get("gradient_accumulation_steps"),
        "lr": training.get("learning_rate"),
        "lr_scheduler_type": training.get("lr_scheduler_type", "cosine"),
        "warmup_ratio": training.get("warmup_ratio", 0.0),
        "weight_decay": training.get("weight_decay", 0.0),
        "max_grad_norm": training.get("max_grad_norm", 1.0),
        "logging_steps": training.get("logging_steps", 5),
        "save_strategy": training.get("save_strategy", "epoch"),
        "save_total_limit": training.get("save_total_limit", 2),
        "eval_strategy": training.get("eval_strategy", "epoch"),
        "load_best_model_at_end": training.get("load_best_model_at_end", True),
        "max_length": training.get("max_seq_length"),
        "eval_ratio": training.get("eval_ratio", 0.10),
        "seed": training.get("seed", 42),
        "fp16": training.get("fp16", False),
        "bf16": training.get("bf16", False),
        "gradient_checkpointing": training.get("gradient_checkpointing", False),
        "lora_r": lora.get("r"),
        "lora_alpha": lora.get("alpha"),
        "lora_dropout": lora.get("dropout"),
        "target_modules": lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        "no_validation": experiment.get("no_validation", False),
    }
    overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "lr": args.lr,
        "max_length": args.max_length,
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    if args.gradient_checkpointing:
        settings["gradient_checkpointing"] = True
    if args.no_validation:
        settings["no_validation"] = True
    required = ["model_name_or_path", "dataset_path", "output_dir", "epochs", "batch_size", "gradient_accumulation", "lr", "max_length", "lora_r", "lora_alpha", "lora_dropout"]
    missing = [key for key in required if settings.get(key) is None]
    if missing:
        raise ValueError("Configuração incompleta; campos ausentes: " + ", ".join(missing))
    return settings


def validate_settings(settings: dict[str, Any]) -> None:
    if settings["epochs"] <= 0 or settings["batch_size"] <= 0 or settings["gradient_accumulation"] <= 0:
        raise ValueError("epochs, batch_size e gradient_accumulation devem ser maiores que zero.")
    if settings["lr"] <= 0 or settings["max_length"] < 32:
        raise ValueError("learning rate deve ser > 0 e max_seq_length deve ser >= 32.")
    if not 0 <= settings["eval_ratio"] < 1:
        raise ValueError("eval_ratio deve estar entre 0 e 1.")
    if settings["lora_r"] <= 0 or settings["lora_alpha"] <= 0:
        raise ValueError("LoRA r e alpha devem ser maiores que zero.")
    if not 0 <= settings["lora_dropout"] < 1:
        raise ValueError("LoRA dropout deve estar entre 0 e 1.")


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
    for package in ("torch", "transformers", "datasets", "peft", "accelerate", "yaml"):
        try:
            __import__(package)
        except ImportError:
            missing.append("pyyaml" if package == "yaml" else package)
    if missing:
        raise RuntimeError("Dependências ausentes: " + ", ".join(missing) + "\nInstale com:\n  pip install torch transformers datasets peft accelerate pyyaml")


def normalize_message(message: Any) -> dict[str, str]:
    if not isinstance(message, dict):
        raise ValueError("cada mensagem deve ser um objeto JSON")
    role, content = message.get("role"), message.get("content")
    if role not in ALLOWED_ROLES:
        raise ValueError(f"role inválida: {role!r}")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("mensagem com conteúdo vazio ou não textual")
    return {"role": role, "content": content.strip()}


def validate_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("'messages' deve conter pelo menos dois turnos")
    normalized = [normalize_message(message) for message in messages]
    if normalized[0]["role"] != "user" or normalized[-1]["role"] != "assistant":
        raise ValueError("a conversa deve começar com user e terminar com assistant")
    previous = None
    for message in normalized:
        role = message["role"]
        if role in {"user", "assistant"} and role == previous:
            raise ValueError(f"turnos consecutivos do mesmo papel: {role}")
        if role in {"user", "assistant"}:
            previous = role
    return normalized


def load_examples(dataset_path: Path) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset processado não encontrado: {dataset_path}\nExecute primeiro: python scripts/prepare_dataset.py")
    examples, invalid = [], 0
    with dataset_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                messages = validate_messages(item.get("messages") if isinstance(item, dict) else None)
                examples.append({"messages": messages, "line": line_number})
            except (json.JSONDecodeError, ValueError) as exc:
                invalid += 1
                print(f"⚠ Linha {line_number} ignorada: {exc}")
    if not examples:
        raise ValueError("Nenhum exemplo válido foi encontrado.")
    print(f"✓ Exemplos válidos: {len(examples)}")
    if invalid:
        print(f"⚠ Exemplos inválidos: {invalid}")
    return examples


def fingerprint_examples(examples: list[dict[str, Any]]) -> str:
    payload = [example["messages"] for example in examples]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def split_examples(examples: list[dict[str, Any]], settings: dict[str, Any]):
    from datasets import Dataset
    dataset = Dataset.from_list([{"messages": x["messages"]} for x in examples])
    if settings["no_validation"] or settings["eval_ratio"] == 0 or len(dataset) < 10:
        print("ℹ Validação interna: desativada")
        return dataset, None
    split = dataset.train_test_split(test_size=settings["eval_ratio"], seed=settings["seed"], shuffle=True)
    print(f"✓ Split treino: {len(split['train'])}")
    print(f"✓ Split validação: {len(split['test'])}")
    return split["train"], split["test"]


def _as_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def native_assistant_mask(tokenizer, messages):
    try:
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_assistant_tokens_mask=True)
    except (TypeError, ValueError, KeyError, NotImplementedError):
        return None
    if not isinstance(encoded, dict):
        return None
    ids = encoded.get("input_ids")
    mask = encoded.get("assistant_masks", encoded.get("assistant_tokens_mask"))
    if ids is None or mask is None:
        return None
    input_ids, assistant_mask = _as_list(ids), _as_list(mask)
    if len(input_ids) != len(assistant_mask) or not any(assistant_mask):
        return None
    return input_ids, assistant_mask


def fallback_assistant_mask(tokenizer, messages):
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    mask = [0] * len(full_ids)
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        before = tokenizer.apply_chat_template(messages[:index], tokenize=False, add_generation_prompt=True)
        through = tokenizer.apply_chat_template(messages[: index + 1], tokenize=False, add_generation_prompt=False)
        start = len(tokenizer(before, add_special_tokens=False)["input_ids"])
        end = len(tokenizer(through, add_special_tokens=False)["input_ids"])
        for position in range(max(0, start), min(len(full_ids), end)):
            mask[position] = 1
    return list(full_ids), mask


def encode_messages(tokenizer, messages):
    native = native_assistant_mask(tokenizer, messages)
    if native is not None:
        return (*native, "native")
    fallback = fallback_assistant_mask(tokenizer, messages)
    if not any(fallback[1]):
        raise ValueError("nenhum token do assistant foi identificado")
    return (*fallback, "fallback")


def trim_sequence(input_ids, mask, max_length):
    if len(input_ids) <= max_length:
        return input_ids, mask, False
    start = len(input_ids) - max_length
    return input_ids[start:], mask[start:], True


def build_tokenized_dataset(tokenizer, dataset, max_length: int):
    from datasets import Dataset
    rows = []
    stats = {"truncated": 0, "multiturn": 0, "native": 0, "fallback": 0, "skipped": 0, "input_tokens": 0, "loss_tokens": 0}
    for example in dataset:
        try:
            input_ids, assistant_mask, mode = encode_messages(tokenizer, example["messages"])
            input_ids, assistant_mask, was_truncated = trim_sequence(input_ids, assistant_mask, max_length)
            stats["truncated"] += int(was_truncated)
            stats[mode] += 1
            stats["multiturn"] += int(len(example["messages"]) > 2)
            labels = [token if active else -100 for token, active in zip(input_ids, assistant_mask)]
            active = sum(label != -100 for label in labels)
            if len(input_ids) < 2 or active == 0:
                stats["skipped"] += 1
                continue
            rows.append({"input_ids": input_ids, "labels": labels})
            stats["input_tokens"] += len(input_ids)
            stats["loss_tokens"] += active
        except Exception as exc:
            stats["skipped"] += 1
            print(f"⚠ Exemplo ignorado durante tokenização: {exc}")
    if not rows:
        raise ValueError("Nenhum exemplo pôde ser tokenizado.")
    print(f"✓ Exemplos tokenizados: {len(rows)}")
    print(f"✓ Multivoltas: {stats['multiturn']}")
    print(f"✓ Truncados: {stats['truncated']}")
    print(f"✓ Máscara nativa: {stats['native']}")
    print(f"✓ Máscara fallback: {stats['fallback']}")
    print(f"✓ Tokens de entrada: {stats['input_tokens']}")
    print(f"✓ Tokens com loss: {stats['loss_tokens']}")
    print(f"✓ Cobertura de loss: {stats['loss_tokens'] / max(stats['input_tokens'], 1):.2%}")
    if stats["skipped"]:
        print(f"⚠ Ignorados: {stats['skipped']}")
    return Dataset.from_list(rows)


def find_checkpoint(output_path: Path) -> str | None:
    if not output_path.exists():
        return None
    checkpoints = []
    for path in output_path.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))
    return str(max(checkpoints, key=lambda item: item[0])[1]) if checkpoints else None


def make_training_args(TrainingArguments, settings: dict[str, Any], torch):
    signature = inspect.signature(TrainingArguments.__init__).parameters
    use_cuda = torch.cuda.is_available()
    evaluation_enabled = not settings["no_validation"] and settings["eval_ratio"] > 0
    params = {
        "output_dir": settings["output_dir"],
        "num_train_epochs": settings["epochs"],
        "per_device_train_batch_size": settings["batch_size"],
        "gradient_accumulation_steps": settings["gradient_accumulation"],
        "learning_rate": settings["lr"],
        "lr_scheduler_type": settings["lr_scheduler_type"],
        "warmup_ratio": settings["warmup_ratio"],
        "weight_decay": settings["weight_decay"],
        "max_grad_norm": settings["max_grad_norm"],
        "logging_steps": settings["logging_steps"],
        "save_strategy": settings["save_strategy"],
        "save_total_limit": settings["save_total_limit"],
        "seed": settings["seed"],
        "fp16": bool(settings["fp16"] and use_cuda),
        "bf16": bool(settings["bf16"] and use_cuda),
        "report_to": "none",
        "remove_unused_columns": False,
    }
    eval_key = "eval_strategy" if "eval_strategy" in signature else "evaluation_strategy"
    params[eval_key] = settings["eval_strategy"] if evaluation_enabled else "no"
    if evaluation_enabled and settings["load_best_model_at_end"]:
        params["load_best_model_at_end"] = True
        params["metric_for_best_model"] = "eval_loss"
        params["greater_is_better"] = False
    return TrainingArguments(**{key: value for key, value in params.items() if key in signature})


def main() -> None:
    args = parse_args()
    config = load_config()
    settings = build_settings(config, args)
    validate_settings(settings)

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path

    model_path = resolve(settings["model_name_or_path"])
    dataset_path = resolve(settings["dataset_path"])
    output_path = resolve(settings["output_dir"])
    settings["output_dir"] = str(output_path)

    print("=" * 64)
    print("🪁 KITE — SFT + LoRA")
    print("=" * 64)
    print(f"Config: {CONFIG_PATH}")
    print(f"Modelo: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Saída: {output_path}")
    print(f"Épocas: {settings['epochs']} | LR: {settings['lr']} | Max length: {settings['max_length']}")
    print(f"LoRA: r={settings['lora_r']} alpha={settings['lora_alpha']} dropout={settings['lora_dropout']}")
    validation_label = "desativada" if settings["no_validation"] else f"{settings['eval_ratio']:.0%}"
    print(f"Validação: {validation_label}")

    check_dependencies()
    set_seed(settings["seed"])
    examples = load_examples(dataset_path)
    fingerprint = fingerprint_examples(examples)
    train_examples, eval_examples = split_examples(examples, settings)

    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments
    from peft import LoraConfig, TaskType, get_peft_model
    import torch

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = build_tokenized_dataset(tokenizer, train_examples, settings["max_length"])
    eval_dataset = build_tokenized_dataset(tokenizer, eval_examples, settings["max_length"]) if eval_examples is not None else None

    if args.dry_run:
        print("\n✅ Dry-run concluído. Nenhum peso foi atualizado.")
        return

    if args.clean_output and output_path.exists():
        print(f"🧹 Removendo checkpoints anteriores de {output_path}...")
        for checkpoint in output_path.glob("checkpoint-*"):
            if checkpoint.is_dir():
                shutil.rmtree(checkpoint)

    output_path.mkdir(parents=True, exist_ok=True)
    lora_config = LoraConfig(
        r=settings["lora_r"],
        lora_alpha=settings["lora_alpha"],
        lora_dropout=settings["lora_dropout"],
        target_modules=settings["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if settings["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    training_args = make_training_args(TrainingArguments, settings, torch)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding=True),
    }
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    resume_checkpoint = find_checkpoint(output_path) if args.resume else None
    print(f"\n🚀 Iniciando treino{f' a partir de {resume_checkpoint}' if resume_checkpoint else ''}...")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    manifest = {
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "dataset_fingerprint": fingerprint,
        "dataset_examples": len(examples),
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
        "settings": settings,
    }
    with (output_path / "training_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    print(f"\n✅ Treino concluído. Adapter salvo em: {output_path}")


if __name__ == "__main__":
    main()
