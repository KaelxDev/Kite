# Raw datasets

Coloque aqui o dataset bruto usado pelo Kite.

## Arquivo esperado

O `prepare_dataset.py` procura especificamente por:

```text
datasets/raw/kite_conversations_v0.1.jsonl
```

Ao executar:

```text
python scripts/prepare_dataset.py
```

se o arquivo não existir, o script mostrará:

```text
Adicione um arquivo como:
  datasets/raw/kite_conversations_v0.1.jsonl
```

## Formato

Uma conversa por linha, em JSONL:

```json
{"user":"Olá!","assistant":"Olá! Como posso ajudar?"}
```

O preparador converte automaticamente esse formato para `messages` e gera:

```text
datasets/processed/train.jsonl
```

Não altere os arquivos brutos. Registre fonte, licença e data de obtenção em `datasets/README.md` ou nos metadados do dataset.
