# Training

Pipeline de treinamento do Kite.

## Fluxo

1. Carregar `Qwen2.5-0.5B-Instruct` em Transformers.
2. Validar `datasets/processed/train.jsonl`.
3. Aplicar SFT + LoRA com PEFT/TRL.
4. Salvar o adapter em `outputs/`.
5. Avaliar antes/depois.
6. Fazer merge somente quando o resultado for aprovado.

O primeiro treino deve ser pequeno e controlado para validar a pipeline antes de usar um dataset grande.
