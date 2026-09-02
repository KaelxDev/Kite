from pathlib import Path
import subprocess
import sys

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


def run_script(script_path):
    """Executa outro módulo usando o mesmo Python do ambiente atual."""
    path = ROOT / script_path

    if not path.exists():
        print(f"\n✗ Arquivo não encontrado: {path}")
        return

    print(f"\n▶ Executando: {script_path}\n")
    result = subprocess.run([sys.executable, str(path)], cwd=ROOT)

    if result.returncode != 0:
        print(f"\n✗ Processo encerrado com código {result.returncode}.")
    else:
        print("\n✓ Processo concluído.")


def test_model():
    # Mantém o LMM.py já validado como teste oficial de inferência.
    run_script("LMM.py")


def prepare_dataset():
    run_script("scripts/prepare_dataset.py")


def train_lora():
    run_script("training/train.py")


def evaluate_model():
    run_script("evaluation/evaluate.py")


def merge_lora():
    run_script("scripts/merge_lora.py")


def project_info():
    print("""
🪁 Kite

Modelo-base:
  Qwen2.5-0.5B-Instruct

Objetivo:
  Especialização conversacional em Português Brasileiro (PT-BR).

Pipeline:
  Modelo-base → Dataset → SFT + LoRA → Avaliação → Merge

Estado atual:
  ✓ Estrutura criada
  ✓ Inferência local validada
  ✓ Orquestrador principal criado
  ⏳ Dataset de treinamento
  ⏳ Treinamento LoRA
  ⏳ Avaliação comparativa
  ⏳ Merge final
""")


def menu():
    while True:
        print("""
────────────────────────────────────────────
[1] 💬 Testar modelo
[2] 📦 Preparar dataset
[3] 🧠 Treinar LoRA
[4] 📊 Avaliar modelo
[5] 🔗 Fazer merge
[6] ℹ️  Informações do projeto
[7] 🔍 Verificar estrutura
[0] 🚪 Sair
────────────────────────────────────────────
""")

        choice = input("Escolha: ").strip()

        actions = {
            "1": test_model,
            "2": prepare_dataset,
            "3": train_lora,
            "4": evaluate_model,
            "5": merge_lora,
            "6": project_info,
            "7": check_structure,
        }

        if choice == "0":
            print("\n🪁 Encerrando Kite...")
            break

        action = actions.get(choice)
        if action is None:
            print("\n✗ Opção inválida.")
            continue

        try:
            action()
        except KeyboardInterrupt:
            print("\n\n⚠ Processo interrompido pelo usuário.")
        except Exception as exc:
            print(f"\n✗ Erro: {exc}")


def main():
    show_banner()
    check_structure()
    menu()


if __name__ == "__main__":
    main()
