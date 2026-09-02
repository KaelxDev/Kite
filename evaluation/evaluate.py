"""Benchmark comparativo do Qwen base e do Kite LoRA.

Uso:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --max-new-tokens 160
    python evaluation/evaluate.py --category IA
    python evaluation/evaluate.py --limit 10
    python evaluation/evaluate.py --sample --temperature 0.7

Importante:
    evaluation/prompts.txt é uma suite de TESTE independente do treinamento.
    O avaliador nao atribui uma nota automatica de qualidade factual, porque
    factualidade, naturalidade e aderencia semantica exigem avaliacao humana
    ou um juiz externo.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "base_model" / "Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = ROOT / "outputs" / "kite-lora"
PROMPTS_PATH = ROOT / "evaluation" / "prompts.txt"
REPORT_PATH = ROOT / "outputs" / "evaluation_report.json"

DEFAULT_MAX_NEW_TOKENS = 160
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SEED = 42

SYSTEM_PROMPT = (
    "Voce e Kite, um assistente util, direto e natural. "
    "Responda em portugues brasileiro quando a pergunta estiver em portugues."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara Qwen base e Kite LoRA")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--sample", action="store_true", help="Ativa amostragem")
    parser.add_argument("--limit", type=int, default=None, help="Avalia apenas os primeiros N prompts")
    parser.add_argument("--category", type=str, default=None, help="Avalia apenas uma categoria")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed usada no modo sampling")
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    return parser.parse_args()


def load_prompts() -> list[dict[str, str]]:
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError(f"Arquivo de prompts nao encontrado: {PROMPTS_PATH}")

    prompts: list[dict[str, str]] = []
    seen_ids: set[str] = set()

    for line_number, raw_line in enumerate(PROMPTS_PATH.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split("|", 2)
        if len(parts) != 3:
            raise ValueError(
                f"Formato invalido na linha {line_number}. Use: ID|categoria|pergunta"
            )

        prompt_id, category, prompt = (part.strip() for part in parts)
        if not prompt_id or not category or not prompt:
            raise ValueError(f"Campos vazios na linha {line_number}.")
        if prompt_id in seen_ids:
            raise ValueError(f"ID duplicado: {prompt_id}")

        seen_ids.add(prompt_id)
        prompts.append({"id": prompt_id, "category": category, "prompt": prompt})

    if not prompts:
        raise ValueError("Nenhum prompt valido foi encontrado.")

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
            "Dependencias ausentes: " + ", ".join(missing) +
            "\nInstale com:\n  pip install torch transformers peft accelerate"
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


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
                f"Adapter LoRA nao encontrado: {ADAPTER_PATH}\n"
                "Execute o treinamento antes da avaliacao."
            )
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.eval()
    return model


def generate_response(model, tokenizer, prompt: str, args: argparse.Namespace) -> dict:
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
        generation_kwargs.update({
            "temperature": args.temperature,
            "top_p": 0.9,
        })

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
    token_count = int(generated_ids.shape[0])

    return {
        "response": response,
        "latency_seconds": round(elapsed, 3),
        "generated_tokens": token_count,
        "tokens_per_second": round(token_count / elapsed, 3) if elapsed > 0 else 0.0,
        "hit_max_tokens": token_count >= args.max_new_tokens,
        "empty_response": not bool(response),
    }


def repetition_ratio(text: str) -> float:
    words = text.lower().split()
    if len(words) < 8:
        return 0.0
    counts = Counter(words)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return round(repeated / len(words), 3)


def evaluate_model(model_name: str, model, tokenizer, prompts: list[dict[str, str]], args) -> list[dict]:
    results = []
    print(f"\n{'=' * 76}\n🧪 {model_name}\n{'=' * 76}")

    for index, item in enumerate(prompts, 1):
        metrics = generate_response(model, tokenizer, item["prompt"], args)
        result = {
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            **metrics,
            "repetition_ratio": repetition_ratio(metrics["response"]),
        }
        results.append(result)

        flags = []
        if result["hit_max_tokens"]:
            flags.append("TRUNCADO?")
        if result["empty_response"]:
            flags.append("VAZIO")
        if result["repetition_ratio"] >= 0.35:
            flags.append("REPETICAO")
        suffix = f" · {' · '.join(flags)}" if flags else ""

        print(f"\n[{index:02d}/{len(prompts)}] {item['id']} · {item['category']}{suffix}")
        print(f"Pergunta: {item['prompt']}")
        print(f"Resposta: {metrics['response'] or '[resposta vazia]'}")
        print(
            f"Tempo: {metrics['latency_seconds']:.2f}s · "
            f"Tokens: {metrics['generated_tokens']} · "
            f"Tok/s: {metrics['tokens_per_second']:.2f}"
        )

    return results


def build_comparison(base_results: list[dict], kite_results: list[dict]) -> list[dict]:
    kite_by_id = {item["id"]: item for item in kite_results}
    comparison = []

    for base in base_results:
        kite = kite_by_id[base["id"]]
        comparison.append({
            "id": base["id"],
            "category": base["category"],
            "prompt": base["prompt"],
            "base": {key: base[key] for key in (
                "response", "latency_seconds", "generated_tokens",
                "tokens_per_second", "hit_max_tokens", "empty_response",
                "repetition_ratio",
            )},
            "kite_lora": {key: kite[key] for key in (
                "response", "latency_seconds", "generated_tokens",
                "tokens_per_second", "hit_max_tokens", "empty_response",
                "repetition_ratio",
            )},
        })

    return comparison


def average(items: list[dict], key: str) -> float:
    values = [item[key] for item in items]
    return round(sum(values) / len(values), 3) if values else 0.0


def rate(items: list[dict], key: str) -> float:
    if not items:
        return 0.0
    return round(sum(bool(item[key]) for item in items) / len(items), 3)


def summarize_model(results: list[dict]) -> dict:
    return {
        "prompt_count": len(results),
        "average_latency_seconds": average(results, "latency_seconds"),
        "average_generated_tokens": average(results, "generated_tokens"),
        "average_tokens_per_second": average(results, "tokens_per_second"),
        "truncation_rate": rate(results, "hit_max_tokens"),
        "empty_response_rate": rate(results, "empty_response"),
        "average_repetition_ratio": average(results, "repetition_ratio"),
    }


def summarize_by_category(results: list[dict]) -> dict:
    grouped = defaultdict(list)
    for item in results:
        grouped[item["category"]].append(item)

    return {
        category: {
            "prompt_count": len(items),
            "average_latency_seconds": average(items, "latency_seconds"),
            "average_generated_tokens": average(items, "generated_tokens"),
            "average_tokens_per_second": average(items, "tokens_per_second"),
            "truncation_rate": rate(items, "hit_max_tokens"),
            "empty_response_rate": rate(items, "empty_response"),
            "average_repetition_ratio": average(items, "repetition_ratio"),
        }
        for category, items in sorted(grouped.items())
    }


def save_report(args, prompts, base_results, kite_results, comparison) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "project": "Kite",
        "evaluation": {
            "prompt_count": len(prompts),
            "categories": sorted({item["category"] for item in prompts}),
            "test_suite": str(PROMPTS_PATH.relative_to(ROOT)),
            "test_is_training_data": False,
            "deterministic": not args.sample,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature if args.sample else None,
            "top_p": 0.9 if args.sample else None,
        },
        "models": {
            "base": str(MODEL_PATH),
            "kite_lora": str(ADAPTER_PATH),
        },
        "summary": {
            "base": summarize_model(base_results),
            "kite_lora": summarize_model(kite_results),
        },
        "by_category": {
            "base": summarize_by_category(base_results),
            "kite_lora": summarize_by_category(kite_results),
        },
        "comparison": comparison,
        "human_review_checklist": [
            "Fidelidade factual",
            "Aderencia a instrucao",
            "Portugues brasileiro",
            "Naturalidade conversacional",
            "Consistencia de identidade e personalidade",
            "Precisao de conceitos tecnicos",
            "Repeticao e respostas genericas",
            "Alucinacoes e afirmacoes sem base",
            "Generalizacao para perguntas nao vistas no treinamento",
        ],
        "note": (
            "Metricas objetivas nao representam qualidade factual. "
            "A suite prompts.txt é independente do treinamento; o relatorio "
            "deve ser usado para comparar comportamento e orientar avaliacao humana."
        ),
    }

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


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
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit deve ser maior que zero.")

    seed_everything(args.seed)
    check_dependencies()
    prompts = load_prompts()

    if args.category:
        prompts = [item for item in prompts if item["category"].lower() == args.category.lower()]
        if not prompts:
            available = ", ".join(sorted({item["category"] for item in load_prompts()}))
            raise ValueError(f"Categoria nao encontrada: {args.category}. Disponiveis: {available}")

    if args.limit is not None:
        prompts = prompts[:args.limit]

    print("\n╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                         🪁 KITE EVALUATION                                ║")
    print("║                  Qwen Base vs. Kite LoRA                                  ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print(f"\nPrompts: {len(prompts)}")
    print("Suite:   evaluation/prompts.txt (TESTE, fora do treinamento)")
    print(f"Modo:    {'sampling' if args.sample else 'deterministico'}")
    print(f"Limite:  {args.max_new_tokens} tokens por resposta")

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

    print("\n╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                       ✓ AVALIACAO CONCLUIDA                               ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print(f"\nRelatorio salvo em: {args.output}")
    print("\nMetricas objetivas: latencia, tokens, tokens/s, truncamento, respostas vazias e repeticao.")
    print("Qualidade factual e aderencia semantica continuam exigindo revisao humana/juiz externo.")


if __name__ == "__main__":
    main()
