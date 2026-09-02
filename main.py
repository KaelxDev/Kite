from pathlib import Path

ROOT = Path(__file__).resolve().parent

DIRECTORIES = (
    "base_model",
    "datasets/raw",
    "datasets/processed",
    "training",
    "evaluation",
    "scripts",
    "outputs",
    "docs",
)


def show_banner():
    print("""
╔══════════════════════════════════════════╗
║                 🪁 KITE                 ║
║       Qwen2.5-0.5B → LoRA → Kite        ║
╚══════════════════════════════════════════╝
""")


def check_structure():
    print("Estrutura do projeto:\n")
    for directory in DIRECTORIES:
        path = ROOT / directory
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {directory}/")

    model_path = ROOT / "base_model" / "Qwen2.5-0.5B-Instruct"
    if model_path.exists():
        print("\n  ✓ Modelo-base encontrado")
    else:
        print("\n  ! Modelo-base não encontrado em base_model/Qwen2.5-0.5B-Instruct")


def main():
    show_banner()
    check_structure()
    print("\nKite: ambiente de desenvolvimento pronto.")
    print("Próxima etapa: preparar o dataset e configurar o treinamento LoRA.")


if __name__ == "__main__":
    main()
