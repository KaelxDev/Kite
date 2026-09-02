# Raw datasets

Coloque aqui os datasets brutos usados pelo Kite.

## Dataset ativo

A preparação atual usa:

```text
datasets/raw/kite_conversations_v0.3.jsonl
```

O v0.3 é uma camada curada focada em corrigir problemas observados na avaliação do Kite, principalmente:

- não-fabricação e representação correta de limitações;
- LoRA, LLM, SFT, RAG, embeddings, overfitting e generalização;
- raciocínio aritmético e causal básico;
- ciência básica;
- seguimento de instruções e restrições de formato;
- português brasileiro natural;
- código e debugging;
- respostas curtas, médias e longas.

## Formato

Uma conversa por linha, em JSONL:

```json
{"user":"Olá!","assistant":"Olá! Como posso ajudar?"}
```

O `prepare_dataset.py` converte automaticamente esse formato para `messages` e gera:

```text
datasets/processed/train.jsonl
```

A preparação preserva quebras de linha, Markdown e blocos de código para que esses padrões também possam ser aprendidos.

## Histórico

- `kite_conversations_v0.1.jsonl`: primeira versão curada; contém problemas identificados na auditoria anterior e não é mais a entrada ativa.
- `kite_conversations_v0.2.jsonl`: expansão com programação, IA, instruções e formatos variados; também contém exemplos que entraram em conflito com a avaliação.
- `kite_conversations_v0.3.jsonl`: versão curada para atacar diretamente esses conflitos.

Não altere ou substitua datasets históricos sem registrar a mudança. Registre fonte, licença e data de obtenção em `datasets/README.md` ou nos metadados do dataset.
