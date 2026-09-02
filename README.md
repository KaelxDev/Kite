# 🪁 Kite

Pequeno modelo de linguagem experimental focado em conversação em Português Brasileiro (PT-BR).

## Base model

Kite utiliza **Qwen2.5-0.5B-Instruct** como modelo-base.

- Base: `Qwen/Qwen2.5-0.5B-Instruct`
- Parâmetros: ~0,49B
- Formato de treinamento: Transformers / Safetensors
- Objetivo: especialização em PT-BR por fine-tuning com LoRA

O modelo GGUF é destinado à inferência. O treinamento utiliza os pesos do modelo-base em formato Transformers.

## Pipeline

```text
Qwen2.5-0.5B-Instruct
        ↓
   Dataset PT-BR
        ↓
    SFT + LoRA
        ↓
   Kite Adapter
        ↓
      Merge
        ↓
   Kite completo
        ↓
      GGUF
        ↓
    MLC / WebLLM
```

## Estrutura

```text
Kite/
├── main.py
├── base_model/                 # modelo Transformers local (não versionar pesos)
├── datasets/
│   ├── raw/                    # datasets originais
│   └── processed/              # datasets preparados
├── training/
│   ├── train.py
│   └── config.yaml
├── evaluation/
│   ├── evaluate.py
│   └── prompts.txt
├── scripts/
│   ├── prepare_dataset.py
│   └── merge_lora.py
├── outputs/                    # checkpoints/adapters gerados
├── docs/
│   └── MODEL_CARD.md
├── MODEL_CARD.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Estado

🚧 **Em desenvolvimento — baseline funcional.**

O modelo-base já foi carregado localmente com Transformers e testado com geração de texto. O próximo estágio é preparar o dataset e executar o primeiro treinamento LoRA.

## Dataset

Os dados próprios do Kite devem ser colocados em `datasets/raw/`.

Formato recomendado: JSONL com mensagens no formato de chat:

```json
{"messages":[{"role":"user","content":"Oi, Kite!"},{"role":"assistant","content":"Oi! Tudo certo? Como posso ajudar?"}]}
```

Use somente datasets cuja origem e licença sejam conhecidas.

## Licença e atribuição

O projeto preserva a informação de licenciamento do modelo-base. Qwen2.5-0.5B-Instruct é o modelo-base de origem; consulte `MODEL_CARD.md` para proveniência e detalhes do projeto.

## Referências

- Qwen2.5-0.5B-Instruct: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Qwen2.5: https://github.com/QwenLM/Qwen2.5
