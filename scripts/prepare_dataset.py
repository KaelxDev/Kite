"""Prepara o dataset híbrido do Kite para treinamento.

Prioridade das fontes:
1. behavioral_core_v1.0: exemplos comportamentais de alta prioridade;
2. v0.9-hybrid: currículo técnico gerado com controle de diversidade;
3. v0.6-curated + multiturn_v0.1: fallback e preservação do material revisado.

A saída contém somente `messages`, sem metadados de geração.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
PROCESSED = ROOT / "datasets" / "processed"
INPUT_FILES = [
    RAW / "kite_behavior_core_v1.0.jsonl",
    RAW / "kite_conversations_v0.9-hybrid.jsonl",
    RAW / "kite_conversations_v0.6-curated.jsonl",
    RAW / "kite_conversations_multiturn_v0.1.jsonl",
]
OUTPUT_FILE = PROCESSED / "train.jsonl"
ALLOWED_ROLES = {"system", "user", "assistant"}


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


def normalize_messages(messages: object) -> dict:
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
        raise ValueError("conversa sem user")
    if not any(m["role"] == "assistant" for m in normalized):
        raise ValueError("conversa sem assistant")
    if normalized[-1]["role"] != "assistant":
        raise ValueError("conversa deve terminar com assistant")
    return {"messages": normalized}


def normalize_example(data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError("a linha deve conter um objeto JSON")
    if "messages" in data:
        return normalize_messages(data["messages"])
    user = normalize_text(data.get("user"))
    assistant = normalize_text(data.get("assistant"))
    if not user or not assistant:
        raise ValueError("user/assistant ausente ou vazio")
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def fingerprint(example: dict) -> str:
    payload = json.dumps(example, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def question_tokens(example: dict) -> set[str]:
    text = " ".join(
        message["content"]
        for message in example["messages"]
        if message["role"] == "user"
    ).lower()
    return set(re.findall(r"[a-záàâãéêíóôõúç0-9_]+", text))


def near_duplicate(a: dict, b: dict, threshold: float = 0.92) -> bool:
    """Evita perguntas quase iguais sem bloquear conversas legítimas sobre o mesmo tema."""
    ta = question_tokens(a)
    tb = question_tokens(b)
    if not ta or not tb:
        return False
    similarity = len(ta & tb) / len(ta | tb)
    return similarity >= threshold


def main() -> None:
    print("🪁 Kite — preparação do dataset híbrido v1.0\n")

    existing = [path for path in INPUT_FILES if path.exists()]
    if not existing:
        raise FileNotFoundError("Nenhuma fonte de dataset foi encontrada em datasets/raw/")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    valid: list[dict] = []
    seen: set[str] = set()
    question_index: list[dict] = []
    invalid = 0
    duplicates = 0
    near_duplicates = 0

    for path in existing:
        print(f"📦 Fonte: {path.name}")
        with path.open("r", encoding="utf-8") as source:
            for line_number, raw in enumerate(source, start=1):
                if not raw.strip():
                    continue
                try:
                    example = normalize_example(json.loads(raw))
                except (json.JSONDecodeError, ValueError) as exc:
                    invalid += 1
                    print(f"⚠ {path.name}:{line_number} ignorada: {exc}")
                    continue

                key = fingerprint(example)
                if key in seen:
                    duplicates += 1
                    continue

                # O núcleo comportamental é prioridade e pode conter perguntas
                # deliberadamente próximas; a filtragem aproximada é aplicada
                # somente às fontes geradas/redundantes depois dele.
                is_core = path.name == "kite_behavior_core_v1.0.jsonl"
                if not is_core and any(near_duplicate(example, old) for old in question_index):
                    near_duplicates += 1
                    continue

                seen.add(key)
                valid.append(example)
                question_index.append(example)

    if not valid:
        raise ValueError("Nenhum exemplo válido encontrado.")

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="\n") as target:
        for example in valid:
            target.write(json.dumps(example, ensure_ascii=False) + "\n")

    multiturn = sum(len(item["messages"]) > 2 for item in valid)
    single = sum(len(item["messages"]) == 2 for item in valid)
    system = sum(any(m["role"] == "system" for m in item["messages"]) for item in valid)

    print("\n✓ Dataset preparado")
    print(f"✓ Exemplos totais: {len(valid)}")
    print(f"✓ Single-turn: {single}")
    print(f"✓ Multivoltas: {multiturn}")
    print(f"✓ Com system: {system}")
    print(f"✓ Duplicatas exatas removidas: {duplicates}")
    print(f"✓ Quase-duplicatas removidas: {near_duplicates}")
    print(f"⚠ Linhas inválidas: {invalid}")
    print(f"✓ Saída: {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
