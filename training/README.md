# Training

Pipeline de treinamento do Kite.

## Configuração

`training/config.yaml` é a **fonte única da configuração do treino**. O `training/train.py` lê esse arquivo no início da execução.

Os parâmetros de linha de comando (`--epochs`, `--lr`, `--lora-r`, etc.) existem somente como **overrides temporários** para experimentos; eles não mantêm defaults separados no código.

Para carregar a configuração YAML, o ambiente precisa de `PyYAML`:

```bash
pip install pyyaml
```

## Fluxo

1. Carregar `Qwen2.5-0.5B-Instruct` em Transformers.
2. Validar `datasets/processed/train.jsonl`.
3. Aplicar SFT + LoRA com PEFT.
4. Salvar o adapter em `outputs/`.
5. Avaliar antes/depois.
6. Fazer merge somente quando o resultado for aprovado.

### Dry-run

Antes de treinar:

```bash
python training/train.py --dry-run
```

O dry-run carrega `training/config.yaml`, valida o dataset e verifica a tokenização/loss sem atualizar os pesos.

O primeiro treino deve ser pequeno e controlado para validar a pipeline antes de usar um dataset grande.
