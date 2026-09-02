"""Prepare os datasets híbridos do Kite para treinamento."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
PROCESSED = ROOT / "datasets" / "processed"
INPUT_FILES = [RAW / "kite_conversations_v0.9-hybrid.jsonl"]
OUTPUT_FILE = PROCESSED / "train.jsonl"
ALLOWED_ROLES = {"system", "user", "assistant"}


def normalize_text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("campo deve ser texto")
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0].strip(): lines.pop(0)
    while lines and not lines[-1].strip(): lines.pop()
    return "\n".join(lines)


def normalize_messages(messages: object) -> dict:
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("'messages' deve conter pelo menos dois turnos")
    normalized = []
    for message in messages:
        if not isinstance(message, dict): raise ValueError("cada mensagem deve ser um objeto")
        role = message.get("role")
        content = normalize_text(message.get("content"))
        if role not in ALLOWED_ROLES: raise ValueError(f"role inválida: {role!r}")
        if not content: raise ValueError("mensagem com conteúdo vazio")
        normalized.append({"role": role, "content": content})
    if not any(m["role"] == "user" for m in normalized): raise ValueError("conversa sem user")
    if not any(m["role"] == "assistant" for m in normalized): raise ValueError("conversa sem assistant")
    if normalized[-1]["role"] != "assistant": raise ValueError("conversa deve terminar com assistant")
    return {"messages": normalized}


def normalize_example(data: object) -> dict:
    if not isinstance(data, dict): raise ValueError("a linha deve conter um objeto JSON")
    if "messages" in data: return normalize_messages(data["messages"])
    user = normalize_text(data.get("user")); assistant = normalize_text(data.get("assistant"))
    if not user or not assistant: raise ValueError("user/assistant ausente ou vazio")
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}


def fingerprint(example: dict) -> str:
    return hashlib.sha256(json.dumps(example, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def main() -> None:
    print("🪁 Kite — Preparação do dataset híbrido v0.9\n")
    existing = [p for p in INPUT_FILES if p.exists()]
    if not existing:
        print("✗ Dataset v0.9 não encontrado. Execute: python scripts/generate_kite_dataset_v0.9.py")
        return
    PROCESSED.mkdir(parents=True, exist_ok=True)
    valid, seen = [], set(); invalid = duplicates = 0
    for path in existing:
        with path.open("r", encoding="utf-8") as source:
            for line_number, raw in enumerate(source, start=1):
                if not raw.strip(): continue
                try: example = normalize_example(json.loads(raw))
                except (json.JSONDecodeError, ValueError) as exc:
                    invalid += 1; print(f"⚠ {path.name}:{line_number} ignorada: {exc}"); continue
                key = fingerprint(example)
                if key in seen: duplicates += 1; continue
                seen.add(key); valid.append(example)
    if not valid: raise ValueError("Nenhum exemplo válido encontrado.")
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="\n") as target:
        for example in valid: target.write(json.dumps(example, ensure_ascii=False) + "\n")
    multiturn = sum(len(x["messages"]) > 2 for x in valid)
    single = sum(len(x["messages"]) == 2 for x in valid)
    print(f"✓ Exemplos totais: {len(valid)}")
    print(f"✓ Single-turn: {single}")
    print(f"✓ Multivoltas: {multiturn}")
    print(f"✓ Duplicatas removidas: {duplicates}")
    print(f"⚠ Linhas inválidas: {invalid}")
    print(f"✓ Saída: {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__": main()
