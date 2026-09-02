"""Dataset preparation entrypoint placeholder."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
PROCESSED = ROOT / "datasets" / "processed"


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW.glob("*.jsonl"))
    print(f"Datasets JSONL encontrados em raw/: {len(files)}")
    print("A preparação será habilitada depois de validarmos o formato dos dados.")


if __name__ == "__main__":
    main()
