"""Prepare o dataset Kite v0.7 curado para treinamento.

Entrada principal:
    datasets/raw/kite_conversations_v0.7-curated.jsonl

Se o arquivo ainda não existir, o script chama automaticamente o gerador
v0.7 para criar uma versão com pelo menos 500 exemplos.

Cada linha deve ser um objeto JSON no formato:
    {"user": "...", "assistant": "..."}

O script preserva quebras de linha, Markdown e blocos de código,
remove duplicatas exatas e converte tudo para o formato `messages`.
"""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
PROCESSED = ROOT / "datasets" / "processed"
INPUT_FILE = RAW / "kite_conversations_v0.7-curated.jsonl"
GENERATOR = ROOT / "scripts" / "generate_kite_dataset_v0.7.py"
OUTPUT_FILE = PROCESSED / "train.jsonl"


def normalize_text(value: object) -> str:
    """Normaliza finais de linha e espaços sem destruir a formatação."""
    if not isinstance(value, str):
        raise ValueError("campo deve ser texto")

    text = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines)


def normalize_example(data: object) -> dict:
    """Valida e converte uma conversa raw para o formato messages."""
    if not isinstance(data, dict):
        raise ValueError("a linha deve conter um objeto JSON")

    user = normalize_text(data.get("user"))
    assistant = normalize_text(data.get("assistant"))

    if not user:
        raise ValueError("campo 'user' vazio ou ausente")
    if not assistant:
        raise ValueError("campo 'assistant' vazio ou ausente")

    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def fingerprint(example: dict) -> str:
    payload = json.dumps(example, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_v07_dataset() -> None:
    """Gera o v0.7 automaticamente quando ele ainda não existir."""
    if INPUT_FILE.exists():
        return

    if not GENERATOR.exists():
        raise FileNotFoundError(
            "Dataset v0.7 e gerador não encontrados. "
            f"Esperado: {GENERATOR}"
        )

    print("⚠ Dataset v0.7 ainda não existe. Gerando automaticamente...\n")
    runpy.run_path(str(GENERATOR), run_name="__main__")

    if not INPUT_FILE.exists():
        raise RuntimeError("O gerador terminou, mas o arquivo v0.7 não foi criado.")


def main() -> None:
    print("🪁 Kite — Preparação do dataset v0.7 curado\n")

    ensure_v07_dataset()
    PROCESSED.mkdir(parents=True, exist_ok=True)

    valid = []
    invalid = 0
    duplicates = 0
    seen = set()

    with INPUT_FILE.open("r", encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                continue

            try:
                example = normalize_example(json.loads(raw_line))
            except (json.JSONDecodeError, ValueError) as exc:
                invalid += 1
                print(f"⚠ Linha {line_number} ignorada: {exc}")
                continue

            key = fingerprint(example)
            if key in seen:
                duplicates += 1
                continue

            seen.add(key)
            valid.append(example)

    if len(valid) < 500:
        raise RuntimeError(
            f"Dataset preparado com apenas {len(valid)} exemplos válidos; "
            "o mínimo esperado é 500."
        )

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="\n") as target:
        for example in valid:
            target.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"✓ Arquivo lido: {INPUT_FILE.relative_to(ROOT)}")
    print(f"✓ Exemplos válidos: {len(valid)}")
    print(f"✓ Duplicatas removidas: {duplicates}")
    print(f"⚠ Linhas inválidas: {invalid}")
    print(f"✓ Dataset processado: {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
