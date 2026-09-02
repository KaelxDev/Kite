"""Evaluation entrypoint placeholder for comparing base and Kite models."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "evaluation" / "prompts.txt"


def main():
    if not PROMPTS.exists():
        raise FileNotFoundError(PROMPTS)

    prompts = [line.strip() for line in PROMPTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Prompts carregados: {len(prompts)}")
    print("Avaliação automática será implementada após o primeiro treinamento.")


if __name__ == "__main__":
    main()
