"""Prepare os datasets brutos curados para o treinamento do Kite.

Entrada:
    datasets/raw/kite_conversations_v0.3.jsonl
    datasets/raw/kite_conversations_v0.4.jsonl

Cada linha deve ser um objeto JSON no formato:
    {"user": "...", "assistant": "..."}

O script combina as versões, preserva Markdown e blocos de código,
remove duplicatas exatas e converte tudo para o formato messages.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
PROCESSED = ROOT / "datasets" / "processed"
INPUT_FILES = (
    RAW / "kite_conversations_v0.3.jsonl",
    RAW / "kite_conversations_v0.4.jsonl",
)
OUTPUT_FILE = PROCESSED / "train.jsonl"


def normalize_text(value: object) -> str:
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
    if not isinstance(data, dict):
        raise ValueError("a linha deve conter um objeto JSON")
    user = normalize_text(data.get("user"))
    assistant = normalize_text(data.get("assistant"))
    if not user:
        raise ValueError("campo 'user' vazio ou ausente")
    if not assistant:
        raise ValueError("campo 'assistant' vazio ou ausente")
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def fingerprint(example: dict) -> str:
    payload = json.dumps(example, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> None:
    print("🪁 Kite — Preparação dos datasets\n")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    valid = []
    invalid = 0
    duplicates = 0
    missing = 0
    seen = set()

    for input_file in INPUT_FILES:
        if not input_file.exists():
            missing += 1
            print(f"⚠ Dataset não encontrado: {input_file.relative_to(ROOT)}")
            continue
        with input_file.open("r", encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    continue
                try:
                    example = normalize_example(json.loads(raw_line))
                except (json.JSONDecodeError, ValueError) as exc:
                    invalid += 1
                    print(f"⚠ {input_file.name}:{line_number} ignorada: {exc}")
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

    print(f"\n✓ Exemplos válidos: {len(valid)}")
    print(f"✓ Duplicatas removidas: {duplicates}")
    print(f"⚠ Linhas inválidas: {invalid}")
    print(f"⚠ Arquivos ausentes: {missing}")
    print(f"✓ Dataset processado: {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
