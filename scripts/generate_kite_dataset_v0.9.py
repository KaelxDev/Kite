"""Gera o dataset híbrido do Kite v0.9.

P0: diversidade semântica e redução de padrões mecânicos.

A versão anterior gerava quatro perguntas quase idênticas para cada conceito.
Isso foi substituído por um banco de tarefas com objetivos diferentes:
- explicação curta;
- contraste entre conceitos;
- diagnóstico de erro;
- decisão de projeto;
- aplicação prática;
- verificação de entendimento.

A geração é determinística, preserva os datasets curados e usa deduplicação
exata. O alvo é 520 exemplos, mas a seleção não corta uma sequência gerada no
meio de um bloco de conceito: os candidatos são embaralhados de forma
reprodutível e escolhidos por uma cota equilibrada entre famílias de tarefa.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
OUT = RAW / "kite_conversations_v0.9-hybrid.jsonl"
BASE = RAW / "kite_conversations_v0.6-curated.jsonl"
MULTI = RAW / "kite_conversations_multiturn_v0.1.jsonl"
TARGET = 520
SEED = 20260902

# O conteúdo factual é curto de propósito: a tarefa deve ensinar a ideia,
# enquanto o modelo aprende diferentes formas de chegar à mesma resposta.
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

TASK_FAMILIES = ("explain", "contrast", "diagnose", "decision", "apply", "verify")

PROMPTS = {
    "explain": [
        "Explique {concept} para alguém que está começando em programação ou IA.",
        "Qual é a ideia central de {concept}? Responda de forma objetiva.",
        "Quero entender {concept} sem uma explicação excessivamente acadêmica. Como você explicaria?",
    ],
    "contrast": [
        "Qual é a diferença entre {concept} e um conceito próximo? Dê um exemplo para deixar a distinção clara.",
        "Estou confundindo {concept} com outra ideia parecida. Como separo corretamente os dois conceitos?",
        "Que erro conceitual alguém poderia cometer ao comparar {concept} com uma técnica ou conceito relacionado?",
    ],
    "diagnose": [
        "Um colega descreveu {concept} de uma forma que parece errada. O que você corrigiria e por quê?",
        "Considere uma implementação que usa {concept}, mas apresenta um resultado inesperado. O que eu deveria verificar primeiro?",
        "Qual é um diagnóstico simples para descobrir se {concept} está sendo entendido ou aplicado incorretamente?",
    ],
    "decision": [
        "Em que situação de um projeto real faz sentido escolher ou considerar {concept}?",
        "Tenho um projeto pequeno e preciso decidir se {concept} é necessário. Que critério devo usar?",
        "Qual trade-off devo considerar antes de adotar {concept} em um sistema?",
    ],
    "apply": [
        "Dê um exemplo prático de {concept} em um projeto de software ou IA.",
        "Como {concept} apareceria no dia a dia de quem desenvolve uma aplicação?",
        "Transforme a definição de {concept} em uma situação concreta de projeto.",
    ],
    "verify": [
        "Como eu verificaria, por teste ou observação, se {concept} está funcionando ou sendo usado corretamente?",
        "Que evidência seria suficiente para dizer que entendi {concept}, em vez de apenas decorar a definição?",
        "Quais sinais indicariam que uma resposta sobre {concept} está correta e não apenas parece convincente?",
    ],
}

ANSWER_SUFFIXES = {
    "explain": "Em termos simples, o ponto principal é: {fact}",
    "contrast": "A distinção deve ser feita pelo significado e pelo comportamento, não apenas pelo nome. {fact}",
    "diagnose": "Comece comparando o comportamento observado com a definição e com a documentação da ferramenta. {fact}",
    "decision": "A escolha depende do problema, das restrições e do custo da solução; não é uma decisão automática. {fact}",
    "apply": "Na prática, isso aparece quando o sistema precisa lidar diretamente com essa propriedade. {fact}",
    "verify": "Uma boa verificação combina definição, comportamento observado e evidência reproduzível. {fact}",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def normalize(row: dict) -> dict | None:
    if "messages" in row and isinstance(row["messages"], list):
        messages = row["messages"]
        if len(messages) >= 2 and messages[-1].get("role") == "assistant":
            clean = []
            for message in messages:
                if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
                    return None
                content = str(message.get("content", "")).strip()
                if not content:
                    return None
                clean.append({"role": message["role"], "content": content})
            return {"messages": clean}
    if row.get("user") and row.get("assistant"):
        return {
            "messages": [
                {"role": "user", "content": str(row["user"]).strip()},
                {"role": "assistant", "content": str(row["assistant"]).strip()},
            ]
        }
    return None


def key(row: dict) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_candidates() -> list[tuple[str, dict]]:
    rng = random.Random(SEED)
    candidates = []
    for concept, fact in CURRICULUM:
        for family in TASK_FAMILIES:
            prompt = rng.choice(PROMPTS[family]).format(concept=concept)
            answer = ANSWER_SUFFIXES[family].format(fact=fact)
            row = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}
            candidates.append((family, row))
    rng.shuffle(candidates)
    return candidates


def select_balanced(candidates: list[tuple[str, dict]], existing: int) -> list[dict]:
    needed = max(0, TARGET - existing)
    if needed == 0:
        return []

    by_family = {family: [] for family in TASK_FAMILIES}
    for family, row in candidates:
        by_family[family].append(row)

    # Primeiro garante distribuição equilibrada entre as seis tarefas.
    selected = []
    seen = set()
    quota = needed // len(TASK_FAMILIES)
    remainder = needed % len(TASK_FAMILIES)

    for index, family in enumerate(TASK_FAMILIES):
        amount = quota + (1 if index < remainder else 0)
        for row in by_family[family]:
            digest = key(row)
            if digest in seen:
                continue
            seen.add(digest)
            selected.append(row)
            if sum(1 for r in selected if key(r) in {key(x) for x in by_family[family]}) >= amount:
                break

    # O cálculo acima é deliberadamente conservador; completa a cota caso
    # alguma família tenha perdido exemplos por deduplicação.
    if len(selected) < needed:
        for family in TASK_FAMILIES:
            for row in by_family[family]:
                digest = key(row)
                if digest in seen:
                    continue
                seen.add(digest)
                selected.append(row)
                if len(selected) == needed:
                    return selected
    return selected[:needed]


def main() -> None:
    rows = []
    seen = set()

    # Dados já curados têm prioridade e não são substituídos pela geração.
    for path in (BASE, MULTI):
        for raw in read_jsonl(path):
            row = normalize(raw)
            if row is None:
                continue
            digest = key(row)
            if digest not in seen:
                seen.add(digest)
                rows.append(row)

    if len(rows) > TARGET:
        raise RuntimeError(
            f"Fontes curadas já possuem {len(rows)} exemplos; alvo configurado: {TARGET}."
        )

    generated = select_balanced(build_candidates(), len(rows))
    for row in generated:
        digest = key(row)
        if digest not in seen:
            seen.add(digest)
            rows.append(row)

    if len(rows) != TARGET:
        raise RuntimeError(f"Dataset ficou com {len(rows)} exemplos; esperado exatamente {TARGET}.")

    OUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )

    single = sum(len(row["messages"]) == 2 for row in rows)
    multi = sum(len(row["messages"]) > 2 for row in rows)
    generated_count = len(generated)
    print(f"Dataset v0.9: {len(rows)} exemplos")
    print(f"Fontes curadas: {len(rows) - generated_count}")
    print(f"Gerados: {generated_count}")
    print(f"Single-turn: {single}")
    print(f"Multivoltas: {multi}")
    print(f"Famílias geradas: {', '.join(TASK_FAMILIES)}")
    print(f"Seed: {SEED}")
    print(f"Saída: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
