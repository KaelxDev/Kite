"""Prepare o dataset bruto para o treinamento do Kite.

Entrada:
    datasets/raw/kite_conversations_v0.3.jsonl

Cada linha deve ser um objeto JSON no formato:
    {"user": "...", "assistant": "..."}

O script converte para:
    {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

A normalização preserva quebras de linha, Markdown e blocos de código.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
PROCESSED = ROOT / "datasets" / "processed"
INPUT_FILE = RAW / "kite_conversations_v0.3.jsonl"
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


def normalize_example(data: object, line_number: int) -> dict:
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


def main() -> None:
    print("🪁 Kite — Preparação do dataset\n")

    if not INPUT_FILE.exists():
        print("✗ Dataset de entrada não encontrado.")
        print()
        print("Adicione o arquivo:")
        print("  datasets/raw/kite_conversations_v0.3.jsonl")
        print()
        print("Formato esperado por linha:")
        print('  {"user":"Olá!","assistant":"Olá! Como posso ajudar?"}')
        return

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
                data = json.loads(raw_line)
                example = normalize_example(data, line_number)
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
