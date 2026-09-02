"""LoRA/SFT training entrypoint.

This file intentionally starts as a validation scaffold. The full training
pipeline will be added after the dataset format is validated.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    model_path = ROOT / "base_model" / "Qwen2.5-0.5B-Instruct"
    dataset_path = ROOT / "datasets" / "processed" / "train.jsonl"
    output_path = ROOT / "outputs" / "kite-lora"

    print("Kite LoRA training")
    print(f"Modelo:  {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Saída:   {output_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo-base não encontrado: {model_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset processado não encontrado: {dataset_path}\n"
            "Coloque um train.jsonl em datasets/processed/ antes de treinar."
        )

    output_path.mkdir(parents=True, exist_ok=True)
    print("\nEstrutura validada. Pipeline de treino será habilitada na próxima etapa.")


if __name__ == "__main__":
    main()
