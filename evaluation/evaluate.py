"""Avalia o Qwen base e o Kite LoRA com os mesmos prompts.

Uso:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --max-new-tokens 120
    python evaluation/evaluate.py --temperature 0.7 --sample

O avaliador foi projetado para comparação qualitativa e reprodutível.
Ele não tenta atribuir uma "nota de qualidade" automaticamente, porque
factualidade, naturalidade e aderência à instrução exigem avaliação humana
ou um juiz externo.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "base_model" / "Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = ROOT / "outputs" / "kite-lora"
PROMPTS_PATH = ROOT / "evaluation" / "prompts.txt"
REPORT_PATH = ROOT / "outputs" / "evaluation_report.json"

DEFAULT_MAX_NEW_TOKENS = 120
DEFAULT_TEMPERATURE = 0.7

SYSTEM_PROMPT = (
    "Você é Kite, um assistente útil, direto e natural. "
    "Responda em português brasileiro quando a pergunta estiver em português."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara Qwen base e Kite LoRA")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Máximo de tokens gerados por resposta",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperatura usada quando --sample estiver ativo",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Ativa amostragem; sem esta opção a geração é determinística",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_PATH,
        help="Arquivo JSON para salvar o relatório",
    )
    return parser.parse_args()


def load_prompts() -> list[dict[str, str]]:
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"Arquivo de prompts não encontrado: {PROMPTS_PATH}")

    prompts: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for line_number, raw_line in enumerate(
        PROMPTS_PATH.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("|", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Formato inválido em prompts.txt na linha {line_number}. "
                "Use: ID|categoria|pergunta"
            )

        prompt_id, category, prompt = (part.strip() for part in parts)
        if not prompt_id or not category or not prompt:
            raise ValueError(f"Campos vazios em prompts.txt na linha {line_number}.")
        if prompt_id in seen_ids:
            raise ValueError(f"ID duplicado em prompts.txt: {prompt_id}")

        seen_ids.add(prompt_id)
        prompts.append({"id": prompt_id, "category": category, "prompt": prompt})

    if not prompts:
        raise ValueError("Nenhum prompt válido foi encontrado.")

    return prompts


def check_dependencies() -> None:
    missing = []
    for package in ("torch", "transformers", "peft"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        raise RuntimeError(
            "Dependências ausentes: " + ", ".join(missing) +
            "\nInstale com:\n"
            "  pip install torch transformers peft accelerate"
        )


def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(use_adapter: bool):
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if use_adapter:
        from peft import PeftModel

        if not ADAPTER_PATH.exists():
            raise FileNotFoundError(
                f"Adapter LoRA não encontrado: {ADAPTER_PATH}\n"
                "Execute o treinamento antes da avaliação."
            )
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.eval()
    return model


def generate_response(model, tokenizer, prompt: str, args: argparse.Namespace) -> tuple[str, float, int]:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.sample:
        generation_kwargs["temperature"] = args.temperature

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        output = model.generate(**inputs, **generation_kwargs)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    generated_ids = output[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    return response, elapsed, int(generated_ids.shape[0])


def evaluate_model(
    model_name: str,
    model,
    tokenizer,
    prompts: list[dict[str, str]],
    args: argparse.Namespace,
) -> list[dict]:
    results = []

    print(f"\n{'=' * 72}")
    print(f"🧪 {model_name}")
    print(f"{'=' * 72}")

    for index, item in enumerate(prompts, 1):
        response, elapsed, token_count = generate_response(
            model, tokenizer, item["prompt"], args
        )

        result = {
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "response": response,
            "latency_seconds": round(elapsed, 3),
            "generated_tokens": token_count,
        }
        results.append(result)

        print(f"\n[{index:02d}/{len(prompts)}] {item['id']} · {item['category']}")
        print(f"Pergunta: {item['prompt']}")
        print(f"Resposta: {response or '[resposta vazia]'}")
        print(f"Tempo: {elapsed:.2f}s · Tokens: {token_count}")

    return results


def build_comparison(base_results: list[dict], kite_results: list[dict]) -> list[dict]:
    kite_by_id = {item["id"]: item for item in kite_results}
    comparison = []

    for base in base_results:
        kite = kite_by_id[base["id"]]
        comparison.append(
            {
                "id": base["id"],
                "category": base["category"],
                "prompt": base["prompt"],
                "base": {
                    "response": base["response"],
                    "latency_seconds": base["latency_seconds"],
                    "generated_tokens": base["generated_tokens"],
                },
                "kite_lora": {
                    "response": kite["response"],
                    "latency_seconds": kite["latency_seconds"],
                    "generated_tokens": kite["generated_tokens"],
                },
            }
        )

    return comparison


def average(items: list[dict], key: str) -> float:
    values = [item[key] for item in items]
    return round(sum(values) / len(values), 3) if values else 0.0


def save_report(args, prompts, base_results, kite_results, comparison) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "project": "Kite",
        "evaluation": {
            "prompt_count": len(prompts),
            "deterministic": not args.sample,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature if args.sample else None,
        },
        "models": {
            "base": str(MODEL_PATH),
            "kite_lora": str(ADAPTER_PATH),
        },
        "summary": {
            "base_average_latency_seconds": average(base_results, "latency_seconds"),
            "kite_average_latency_seconds": average(kite_results, "latency_seconds"),
            "base_average_generated_tokens": average(base_results, "generated_tokens"),
            "kite_average_generated_tokens": average(kite_results, "generated_tokens"),
        },
        "comparison": comparison,
        "note": (
            "O relatório não atribui notas automáticas de qualidade. "
            "Use a comparação das respostas para avaliar factualidade, "
            "aderência à instrução, naturalidade e comportamento conversacional."
        ),
    }

    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def release_model(model) -> None:
    del model
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main() -> None:
    args = parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens deve ser maior que zero.")
    if args.sample and args.temperature <= 0:
        raise ValueError("--temperature deve ser maior que zero.")

    check_dependencies()
    prompts = load_prompts()

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║                    🪁 KITE EVALUATION                              ║")
    print("║              Qwen Base vs. Kite LoRA                               ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\nPrompts: {len(prompts)}")
    print(f"Modelo base: {MODEL_PATH}")
    print(f"Adapter:     {ADAPTER_PATH}")
    print(f"Geração:     {'sampling' if args.sample else 'determinística'}")

    tokenizer = load_tokenizer()

    print("\n📦 Carregando modelo base...")
    base_model = load_model(use_adapter=False)
    base_results = evaluate_model("QWEN BASE", base_model, tokenizer, prompts, args)
    release_model(base_model)

    print("\n📦 Carregando modelo + adapter Kite...")
    kite_model = load_model(use_adapter=True)
    kite_results = evaluate_model("KITE LoRA", kite_model, tokenizer, prompts, args)

    comparison = build_comparison(base_results, kite_results)
    save_report(args, prompts, base_results, kite_results, comparison)
    release_model(kite_model)

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║                    ✓ AVALIAÇÃO CONCLUÍDA                           ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\nRelatório salvo em: {args.output}")
    print("\nCritérios para análise humana:")
    print("  • Fidelidade factual")
    print("  • Aderência à instrução")
    print("  • Português brasileiro")
    print("  • Naturalidade conversacional")
    print("  • Consistência de personalidade")
    print("  • Repetição / respostas genéricas")
    print("  • Alucinações")


if __name__ == "__main__":
    main()
