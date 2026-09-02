"""Gera o dataset híbrido do Kite v0.9.

Objetivo: aumentar cobertura sem inflar o dataset com simples paráfrases.
Fontes preservadas:
- v0.6-curated: base single-turn
- multiturn_v0.1: conversas multivoltas

As expansões abaixo mudam a tarefa (definição, diagnóstico, decisão ou
aplicação), em vez de apenas trocar palavras da mesma pergunta.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
OUT = RAW / "kite_conversations_v0.9-hybrid.jsonl"
BASE = RAW / "kite_conversations_v0.6-curated.jsonl"
MULTI = RAW / "kite_conversations_multiturn_v0.1.jsonl"
TARGET = 520

# Cada item descreve uma ideia. As quatro tarefas geradas têm objetivos
# diferentes e, portanto, não são simples paráfrases.
CURRICULUM = [
("LLM", "Um LLM aprende padrões estatísticos de linguagem e os usa para estimar ou gerar sequências de tokens."),
("token", "Um token é uma unidade da representação textual usada pelo modelo; sua granularidade depende do tokenizador."),
("context window", "A janela de contexto limita a quantidade de tokens que o modelo pode considerar em uma entrada."),
("temperature", "Temperatura altera a distribuição usada na amostragem dos próximos tokens e pode aumentar ou reduzir a aleatoriedade."),
("top-p", "Top-p restringe a amostragem a um conjunto de tokens cuja probabilidade acumulada atinge um limite definido."),
("quantização", "Quantização representa parâmetros ou ativações com menor precisão numérica para reduzir memória e custo."),
("inference", "Inferência é o uso de um modelo já treinado para produzir uma saída a partir de uma entrada."),
("checkpoint", "Checkpoint é uma versão salva do estado do treinamento que pode ser usada para avaliação ou continuação."),
("seed", "Seed inicializa fontes de aleatoriedade e pode ajudar na reprodutibilidade, embora não elimine toda diferença entre execuções."),
("perplexidade", "Perplexidade mede quão bem um modelo prevê os tokens observados, mas não captura sozinha todas as qualidades de um chatbot."),
("fine-tuning", "Fine-tuning é treinamento adicional de um modelo pré-treinado para adaptá-lo a uma tarefa, domínio ou comportamento."),
("SFT", "SFT é Supervised Fine-Tuning e usa exemplos supervisionados de entrada e resposta-alvo."),
("PEFT", "PEFT reúne técnicas que adaptam um modelo treinando apenas uma fração dos parâmetros."),
("LoRA", "LoRA usa adaptações de baixa ordem e normalmente mantém os pesos originais do modelo-base congelados."),
("rank LoRA", "O rank determina a dimensão da representação de baixa ordem e influencia capacidade e quantidade de parâmetros treináveis."),
("lora_alpha", "lora_alpha controla a escala da contribuição do caminho LoRA conforme a implementação."),
("lora_dropout", "lora_dropout aplica dropout no caminho do adapter durante o treinamento quando essa regularização é configurada."),
("target_modules", "target_modules define em quais módulos da arquitetura os adapters serão inseridos."),
("adapter", "Um adapter LoRA contém parâmetros adicionais aprendidos para adaptar um modelo-base."),
("merge", "Merge incorpora a contribuição do adapter aos pesos do modelo-base, criando uma versão integrada."),
("RAG", "RAG recupera informações relevantes e fornece esse contexto ao modelo durante a geração."),
("embedding", "Embedding é uma representação vetorial usada, entre outras coisas, para comparar itens por similaridade."),
("vector search", "Busca vetorial recupera itens comparando representações vetoriais segundo uma medida de similaridade."),
("hybrid search", "Busca híbrida combina sinais diferentes, como busca lexical e semântica, para melhorar a recuperação."),
("chunking", "Chunking divide documentos em trechos menores para facilitar indexação e recuperação de contexto."),
("reranking", "Reranking reordena candidatos recuperados usando um critério ou modelo mais preciso."),
("overfitting", "Overfitting ocorre quando o modelo se adapta excessivamente ao treino e perde desempenho em dados novos."),
("generalização", "Generalização é a capacidade de manter bom desempenho em exemplos não usados diretamente no ajuste."),
("data leakage", "Data leakage ocorre quando informações que deveriam estar isoladas entram indevidamente no treinamento ou ajuste."),
("train/validation/test", "Treino ajusta parâmetros; validação orienta decisões durante desenvolvimento; teste fornece avaliação final independente."),
("dataset quality", "Qualidade do dataset depende de correção, consistência, relevância, diversidade e cobertura, não apenas de quantidade."),
("duplicate data", "Duplicatas excessivas reduzem diversidade efetiva e podem dar peso desproporcional a determinados padrões."),
("data augmentation", "Data augmentation cria novas variações úteis sem depender de cópias quase idênticas dos exemplos existentes."),
("assistant-only loss", "Em SFT conversacional, calcular loss apenas nas respostas do assistente alinha o objetivo com o comportamento que se quer ensinar."),
("multiturn", "Exemplos multivoltas treinam respostas que dependem de contexto acumulado dentro da conversa."),
("system message", "Uma mensagem system representa instruções ou contexto global que pode orientar a conversa quando o template suporta esse papel."),
("ambiguity", "Quando o contexto não resolve uma ambiguidade relevante, o modelo deve pedir esclarecimento ou explicitar as interpretações."),
("non-fabrication", "Quando faltam evidências, o modelo deve declarar a incerteza e não apresentar uma invenção como fato."),
("correction", "Uma correção do usuário deve ser considerada, mas não aceita automaticamente se contradizer evidências confiáveis."),
("confidence", "Fluência ou confiança textual não é prova de correção factual."),
("API", "API é uma interface que define como componentes de software podem se comunicar."),
("endpoint", "Endpoint é um ponto específico de uma API associado a uma operação ou recurso."),
("HTTP 404", "HTTP 404 indica que o servidor não encontrou o recurso solicitado."),
("HTTP 500", "HTTP 500 indica um erro interno do servidor e não revela sozinho sua causa."),
("REST", "REST é um estilo arquitetural para sistemas distribuídos que enfatiza recursos, representações e operações por interfaces uniformes."),
("JSON", "JSON é um formato textual para representar dados estruturados e não uma linguagem de programação."),
("authentication", "Autenticação verifica a identidade de uma entidade."),
("authorization", "Autorização determina quais recursos ou ações uma identidade autenticada pode acessar."),
("JWT", "JWT é um formato de token que pode transportar claims; seu uso não substitui autenticação, autorização e validações adequadas."),
("password hashing", "Senhas devem ser protegidas com funções de hashing apropriadas para senha, com salt e parâmetros adequados."),
("CORS", "CORS controla, no navegador, quais origens podem fazer determinadas requisições cross-origin conforme os cabeçalhos da resposta."),
("SQL injection", "SQL injection ocorre quando entrada não confiável altera indevidamente a estrutura de uma consulta SQL; consultas parametrizadas ajudam a evitar isso."),
("input validation", "Validação de entrada verifica se dados recebidos obedecem às regras esperadas antes de serem processados."),
("rate limiting", "Rate limiting limita a quantidade de requisições permitidas em determinado período."),
("timeout", "Timeout define um limite para esperar uma operação antes de tratá-la como excedida."),
("logging", "Logging registra eventos da aplicação para diagnóstico e observabilidade, sem expor segredos desnecessários."),
("environment variable", "Variáveis de ambiente fornecem configuração ao processo sem exigir que valores sejam codificados diretamente no código."),
("dependency pinning", "Fixar versões de dependências ajuda a reproduzir ambientes e reduzir mudanças inesperadas."),
("unit test", "Teste unitário verifica uma unidade de código de forma isolada."),
("integration test", "Teste de integração verifica a interação entre componentes ou serviços."),
("mock", "Mock substitui uma dependência real por um comportamento controlado durante um teste."),
("CI", "Continuous Integration automatiza integração frequente de mudanças e verificações como testes e lint."),
("CD", "Continuous Delivery ou Deployment automatiza partes da entrega de software, dependendo da definição adotada."),
("Git", "Git é um sistema distribuído de controle de versões."),
("GitHub", "GitHub é uma plataforma que hospeda repositórios Git e fornece recursos de colaboração e automação."),
("branch", "Branch é uma linha de desenvolvimento independente dentro do histórico do Git."),
("commit", "Commit registra um conjunto de alterações no histórico de um repositório Git."),
("rebase", "Rebase reaplica commits sobre outra base e pode reorganizar o histórico de uma branch."),
("Docker", "Docker fornece uma forma de empacotar e executar aplicações em containers, que normalmente compartilham o kernel do host."),
("cache", "Cache armazena dados ou resultados reutilizáveis para reduzir custo ou latência de acessos posteriores."),
("database index", "Índices podem acelerar consultas específicas, mas ocupam espaço e podem aumentar o custo de escritas."),
("primary key", "Chave primária identifica unicamente registros de uma tabela."),
("foreign key", "Chave estrangeira representa uma relação com outra tabela e ajuda a preservar integridade referencial."),
("transaction", "Transação agrupa operações de banco que devem obedecer às propriedades de consistência definidas pelo sistema."),
("ACID", "ACID representa Atomicidade, Consistência, Isolamento e Durabilidade."),
("normalization", "Normalização organiza dados relacionais para reduzir redundância e anomalias de atualização."),
("migration", "Migration é uma alteração versionada do esquema do banco que pode ser aplicada de forma controlada."),
("async/await", "async/await fornece sintaxe para trabalhar com operações assíncronas sem significar automaticamente paralelismo."),
("concurrency", "Concorrência trata do progresso coordenado de múltiplas tarefas, mesmo sem execução simultânea literal."),
("parallelism", "Paralelismo envolve execução simultânea de trabalho em múltiplos recursos."),
("deadlock", "Deadlock ocorre quando tarefas ficam bloqueadas esperando recursos umas das outras e nenhuma consegue prosseguir."),
("Big O", "Big O descreve como o custo de um algoritmo cresce assintoticamente com o tamanho da entrada."),
("refactoring", "Refatoração reorganiza a estrutura interna do código sem alterar intencionalmente seu comportamento externo."),
("technical debt", "Dívida técnica representa custos futuros de manutenção ou risco criados por decisões que priorizam benefícios imediatos."),
("CPU", "CPU executa instruções e operações computacionais do programa."),
("RAM", "RAM é memória de trabalho volátil usada por programas em execução."),
("storage", "Armazenamento como SSD ou HD mantém dados mesmo quando o dispositivo está desligado."),
("photosynthesis", "Fotossíntese usa energia luminosa para produzir compostos orgânicos a partir de dióxido de carbono e água em organismos fotossintéticos."),
("Rayleigh scattering", "O espalhamento de Rayleigh contribui para a aparência azul do céu porque comprimentos de onda menores são espalhados com maior eficiência."),
("black hole", "Buraco negro é uma região do espaço-tempo da qual não é possível escapar para fora do horizonte de eventos sob a descrição relativística."),
("heat vs temperature", "Temperatura caracteriza o estado térmico; calor é energia transferida devido a uma diferença de temperatura."),
("Moon phases", "As fases da Lua resultam da mudança da porção iluminada visível da Terra conforme mudam as posições relativas de Lua, Terra e Sol."),
("natural selection", "Seleção natural altera a frequência de características hereditárias quando diferenças de reprodução estão associadas ao ambiente."),
]


def read_jsonl(path: Path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def normalize(row):
    if "messages" in row and isinstance(row["messages"], list):
        msgs = row["messages"]
        if len(msgs) >= 2 and msgs[-1].get("role") == "assistant":
            return {"messages": [{"role": m.get("role"), "content": str(m.get("content", "")).strip()} for m in msgs]}
    if row.get("user") and row.get("assistant"):
        return {"messages": [{"role": "user", "content": str(row["user"]).strip()}, {"role": "assistant", "content": str(row["assistant"]).strip()}]}
    return None


def key(row):
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main():
    rows = []
    seen = set()
    for path in (BASE, MULTI):
        for raw in read_jsonl(path):
            row = normalize(raw)
            if row and key(row) not in seen:
                seen.add(key(row)); rows.append(row)

    # Quatro tipos de tarefa por conceito: entendimento, decisão, diagnóstico e aplicação.
    for concept, answer in CURRICULUM:
        examples = [
            (f"Explique o conceito de {concept} e destaque sua ideia principal.", answer),
            (f"Estou trabalhando com {concept}. Qual é um erro comum de interpretação que devo evitar?", f"Um erro comum é simplificar demais o conceito. {answer}"),
            (f"Em um projeto real, quando {concept} seria relevante e por quê?", f"{answer} Ele é relevante quando a tarefa depende diretamente dessa propriedade ou comportamento."),
            (f"Como eu verificaria se estou usando {concept} corretamente em uma implementação ou avaliação?", f"Verifique a definição e compare-a com o comportamento observado. {answer}"),
        ]
        for q, a in examples:
            row = {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
            k = key(row)
            if k not in seen:
                seen.add(k); rows.append(row)

    if len(rows) < TARGET:
        raise RuntimeError(f"Dataset ficou com {len(rows)} exemplos; mínimo: {TARGET}")

    rows = rows[:TARGET]
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows), encoding="utf-8")

    single = sum(len(r["messages"]) == 2 for r in rows)
    multi = sum(len(r["messages"]) > 2 for r in rows)
    print(f"Dataset v0.9: {len(rows)} exemplos")
    print(f"Single-turn: {single}")
    print(f"Multivoltas: {multi}")
    print(f"Saída: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
