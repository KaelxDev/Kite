"""Prepare os datasets do Kite para treinamento.

Fontes de treinamento:
    datasets/raw/kite_conversations_v0.7-curated.jsonl
    datasets/raw/kite_conversations_multiturn_v0.1.jsonl

Os dois formatos são aceitos:
    {"user": "...", "assistant": "..."}
    {"messages": [{"role": "user", "content": "..."}, ...]}

O script preserva conversas multivoltas, quebras de linha, Markdown e
blocos de código. Duplicatas exatas são removidas antes da conversão para
`datasets/processed/train.jsonl`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
PROCESSED = ROOT / "datasets" / "processed"

INPUT_FILES = [
    RAW / "kite_conversations_v0.7-curated.jsonl",
    RAW / "kite_conversations_multiturn_v0.1.jsonl",
]
OUTPUT_FILE = PROCESSED / "train.jsonl"

ALLOWED_ROLES = {"system", "user", "assistant"}


def normalize_text(value: object) -> str:
    """Normaliza finais de linha e espaços sem destruir formatação."""
    if not isinstance(value, str):
        raise ValueError("campo deve ser texto")

    text = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines)


def normalize_messages(messages: object) -> dict:
    """Valida e normaliza uma conversa já no formato messages."""
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("'messages' deve conter pelo menos dois turnos")

    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("cada mensagem deve ser um objeto")

        role = message.get("role")
        content = normalize_text(message.get("content"))

        if role not in ALLOWED_ROLES:
            raise ValueError(f"role inválida: {role!r}")
        if not content:
            raise ValueError("mensagem com conteúdo vazio")

        normalized.append({"role": role, "content": content})

    if not any(m["role"] == "user" for m in normalized):
        raise ValueError("conversa sem turno de user")
    if not any(m["role"] == "assistant" for m in normalized):
        raise ValueError("conversa sem turno de assistant")

    # Mensagens consecutivas do mesmo papel são permitidas, mas evitamos
    # estruturas estranhas que indiquem dados malformados.
    return {"messages": normalized}


def normalize_example(data: object) -> dict:
    """Converte os formatos raw suportados para `messages`."""
    if not isinstance(data, dict):
        raise ValueError("a linha deve conter um objeto JSON")

    if "messages" in data:
        return normalize_messages(data["messages"])

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
    print("🪁 Kite — Preparação do dataset de treinamento\n")

    existing_files = [path for path in INPUT_FILES if path.exists()]
    if not existing_files:
        print("✗ Nenhum dataset de entrada encontrado.")
        print()
        for path in INPUT_FILES:
            print(f"  {path.relative_to(ROOT)}")
        print()
        print("O dataset multivoltas recomendado é:")
        print("  datasets/raw/kite_conversations_multiturn_v0.1.jsonl")
        return

    PROCESSED.mkdir(parents=True, exist_ok=True)

    valid: list[dict] = []
    invalid = 0
    duplicates = 0
    seen = set()
    source_counts = {}

    for input_file in existing_files:
        source_valid = 0
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
                source_valid += 1

        source_counts[input_file.name] = source_valid

    if not valid:
        raise ValueError("Nenhum exemplo válido foi encontrado.")

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="\n") as target:
        for example in valid:
            target.write(json.dumps(example, ensure_ascii=False) + "\n")

    print("Arquivos usados:")
    for name, count in source_counts.items():
        print(f"  ✓ {name}: {count} exemplos")

    print(f"\n✓ Exemplos totais válidos: {len(valid)}")
    print(f"✓ Duplicatas removidas: {duplicates}")
    print(f"⚠ Linhas inválidas: {invalid}")
    print(f"✓ Dataset processado: {OUTPUT_FILE.relative_to(ROOT)}")

    multiturn = sum(1 for item in valid if len(item["messages"]) > 2)
    print(f"✓ Conversas multivoltas: {multiturn}")


if __name__ == "__main__":
    main()
