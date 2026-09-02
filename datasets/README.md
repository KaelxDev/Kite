# Datasets

Os datasets usados pelo Kite ficam separados por etapa.

```text
datasets/
├── raw/        # arquivos originais
└── processed/  # dados preparados para treinamento
```

## Formato recomendado

JSONL com mensagens de chat:

```json
{"messages":[{"role":"user","content":"Oi, Kite!"},{"role":"assistant","content":"Oi! Tudo certo? Como posso ajudar?"}]}
```

## Regras

- Preserve a origem de cada dataset.
- Registre a licença e a fonte.
- Não misture arquivos brutos com arquivos processados.
- Remova dados pessoais ou sensíveis antes do treinamento.
