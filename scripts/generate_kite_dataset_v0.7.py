"""Gera o dataset Kite v0.7 com 500+ exemplos curados.

A base v0.6-curated permanece preservada. O script cria variações controladas
sobre exemplos vetados/curados, sem alterar a suíte de teste.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "raw"
BASE_FILE = RAW / "kite_conversations_v0.6-curated.jsonl"
OUTPUT_FILE = RAW / "kite_conversations_v0.7-curated.jsonl"
TARGET = 520
SEED = 42

VARIANTS = [
    "{}",
    "Explique de forma clara: {}",
    "Quero entender melhor: {}",
    "Como você explicaria isso para alguém que está começando? {}",
    "Qual é a ideia central? {}",
]

# Banco de sementes adicional. As respostas foram revisadas para manter
# consistência factual e evitar os erros observados nas versões anteriores.
SEED_TEXT = r'''Quem desenvolveu o Kite?	O Kite foi desenvolvido por Kaike como um projeto experimental de modelo de linguagem.
Qual é o modelo-base usado pelo Kite?	O modelo-base usado pelo Kite é o Qwen2.5-0.5B-Instruct.
O Kite foi treinado do zero?	Não. O Kite é uma especialização construída sobre um modelo-base pré-treinado.
Qual é o objetivo do Kite?	O objetivo do Kite é experimentar a especialização de um modelo pequeno para conversação em Português Brasileiro, com respostas úteis e naturais.
Como o Kite deve agir quando não sabe um fato?	Deve declarar a incerteza, evitar inventar informações e indicar como verificar o fato quando possível.
Um assistente pode afirmar que executou código sem executar?	Não. Deve distinguir entre analisar código e realmente executá-lo.
Um assistente pode dizer que leu um arquivo sem acesso a ele?	Não. Deve informar que o arquivo não foi fornecido ou não pôde ser acessado.
Uma resposta confiante prova que está correta?	Não. Fluência e confiança no texto não são evidência suficiente de correção factual.
Como lidar com uma hipótese sem evidência suficiente?	Deve ser apresentada como hipótese ou possibilidade, não como fato.
O que fazer quando uma ferramenta falha?	Informar que a operação não foi concluída e não afirmar que ela funcionou.
O que significa LLM no contexto de IA?	LLM significa Large Language Model, ou modelo de linguagem de grande escala.
Qual é a função de um modelo de linguagem?	Ele aprende padrões de linguagem a partir de dados e usa esses padrões para estimar ou gerar sequências de tokens.
O que é tokenização?	É o processo de transformar texto em tokens de acordo com o tokenizador do modelo.
O que é um token em um modelo de linguagem?	É uma unidade usada na representação do texto; pode corresponder a uma palavra, parte dela, pontuação ou outro trecho.
Qual é a diferença entre treinamento e inferência?	Treinamento ajusta os parâmetros do modelo com dados; inferência usa o modelo ajustado para produzir uma saída.
O que caracteriza o fine-tuning?	É o treinamento adicional de um modelo pré-treinado para adaptá-lo a uma tarefa, domínio, formato ou comportamento.
O que significa SFT?	SFT significa Supervised Fine-Tuning e usa exemplos de entrada e resposta-alvo para ajustar o comportamento do modelo.
O que significa PEFT?	PEFT significa Parameter-Efficient Fine-Tuning e busca adaptar o modelo treinando apenas uma fração dos parâmetros.
O que significa LoRA em IA?	LoRA significa Low-Rank Adaptation e é uma técnica de PEFT para adaptar um modelo treinando parâmetros adicionais de baixa dimensão.
Os pesos do modelo-base ficam congelados no LoRA convencional?	Sim. No LoRA convencional, os pesos originais normalmente ficam congelados e os parâmetros do adapter são treinados.
O que é um adapter LoRA?	É o conjunto de parâmetros adicionais treinados para representar a adaptação LoRA sobre um modelo-base.
Um adapter LoRA contém uma cópia completa do modelo?	Normalmente não. Ele contém os parâmetros adicionais da adaptação, enquanto os pesos do modelo-base permanecem separados.
O que é rank no LoRA?	É a dimensão de baixa ordem usada para representar a atualização aprendida pelo adapter.
Rank maior sempre melhora o LoRA?	Não. Um rank maior aumenta a capacidade do adapter, mas não garante melhor generalização e pode elevar o custo.
O que é lora_r?	É o parâmetro que define o rank usado pelas matrizes de baixa dimensão do adapter LoRA.
O que é lora_alpha?	É um fator de escala que controla a contribuição da atualização LoRA, conforme a implementação.
O que é lora_dropout?	É a taxa de dropout aplicada ao caminho do adapter durante o treinamento, ajudando na regularização quando usada.
O que são target_modules no LoRA?	São os módulos ou camadas onde os adapters LoRA serão inseridos.
O que é merge de LoRA?	É o processo de incorporar a atualização aprendida pelo adapter aos pesos do modelo-base, criando uma versão integrada.
O merge é obrigatório para usar um adapter LoRA?	Não. O adapter pode ser carregado sobre o modelo-base sem ser mesclado.
LoRA e fine-tuning completo são iguais?	Não. O ajuste completo atualiza diretamente os pesos do modelo; LoRA normalmente treina apenas parâmetros adicionais de baixa dimensão.
LoRA e SFT são a mesma coisa?	Não. SFT descreve um tipo de treinamento supervisionado, enquanto LoRA é uma técnica eficiente de adaptação.
LoRA e LoRa significam a mesma coisa?	Não. Em IA, LoRA significa Low-Rank Adaptation; LoRa também é uma tecnologia de comunicação sem fio.
Como escolher o significado de LoRA quando há contexto?	Use o contexto. Em modelos e fine-tuning, LoRA significa Low-Rank Adaptation; em rádio e IoT, LoRa pode significar a tecnologia de comunicação.
O que é RAG?	RAG significa Retrieval-Augmented Generation e combina recuperação de contexto com geração de texto.
RAG altera os pesos do modelo a cada consulta?	Normalmente não. RAG recupera contexto durante a consulta sem precisar atualizar os pesos do modelo.
RAG substitui fine-tuning?	Não necessariamente. RAG adiciona contexto recuperado; fine-tuning adapta os parâmetros do modelo.
O que é um embedding?	É uma representação vetorial de um item, como texto, que pode ser usada para comparar ou recuperar conteúdo por similaridade.
Embedding e token são a mesma coisa?	Não. Token é uma unidade de representação textual; embedding é uma representação vetorial associada a tokens ou outros objetos, dependendo do modelo.
O que é overfitting?	É quando o modelo se adapta excessivamente aos dados de treinamento e perde desempenho em exemplos novos.
O que é generalização em machine learning?	É a capacidade de um modelo apresentar bom desempenho em dados novos que não foram usados diretamente no ajuste.
O que é data leakage?	É a entrada indevida de informações do conjunto de teste ou validação no treinamento ou nas decisões de ajuste, tornando as métricas artificialmente otimistas.
Por que separar treino, validação e teste?	Treino serve para ajustar o modelo, validação ajuda nas decisões durante o desenvolvimento e teste mede o desempenho final em dados mantidos separados.
Por que não colocar prompts de teste no treinamento?	Porque o modelo pode memorizar exemplos específicos da avaliação, tornando o resultado menos representativo da generalização.
Por que qualidade do dataset importa?	Porque exemplos incorretos, contraditórios ou mal formulados podem ensinar padrões indesejados.
Por que usar paráfrases no dataset?	Para ensinar diferentes formas de perguntar a mesma coisa e reduzir dependência de correspondência literal.
Repetir exemplos é sempre positivo?	Não. Repetição excessiva reduz diversidade e pode favorecer memorização superficial.
Por que preservar Markdown em exemplos de treinamento?	Porque listas, tabelas e blocos de código fazem parte dos formatos que o modelo pode precisar produzir.
Por que incluir exemplos de diferentes comprimentos?	Para ensinar que perguntas simples podem ter respostas curtas, enquanto tutoriais e tarefas complexas exigem mais detalhes.
Por que incluir código no dataset?	Porque ensina o modelo a produzir e analisar soluções práticas, não apenas definições.
Por que incluir debugging?	Porque ensina a identificar erros, explicar causas e propor correções.
Quanto é 18 + 24?	42.
Quanto é 96 / 12?	8.
Quanto é 25% de 200?	50.
Se A é maior que B e B é maior que C, o que concluímos?	A é maior que C.
Duas tarefas independentes de 8 segundos podem rodar em paralelo. Qual é o tempo ideal do conjunto?	Aproximadamente 8 segundos, assumindo execução realmente paralela e sem gargalos adicionais.
Uma tarefa leva 10 segundos e é executada duas vezes em sequência. Quanto tempo leva?	20 segundos.
Um produto de R$ 120 recebe desconto de 25%. Qual é o preço final?	R$ 90.
15 representa qual porcentagem de 20?	75%.
Se 3 de 12 testes falham, qual é a taxa de falha?	25%.
Se uma latência cai de 300 ms para 200 ms, qual foi a redução?	100 ms.
Por que o céu parece azul?	Principalmente pelo espalhamento de Rayleigh, que espalha a luz azul com mais eficiência na atmosfera.
O que é um buraco negro?	É uma região do espaço-tempo cuja gravidade é tão intensa que, dentro do horizonte de eventos, nem a luz consegue escapar.
Por que a água pode evaporar antes de ferver?	Porque algumas moléculas na superfície têm energia suficiente para passar à fase gasosa mesmo abaixo do ponto de ebulição.
Por que a grama geralmente é verde?	Porque a clorofila absorve mais fortemente algumas regiões da luz visível e reflete mais luz na região verde.
O que é gravidade?	É a interação associada à massa e à energia; na relatividade geral, é descrita pela curvatura do espaço-tempo.
O que é fotossíntese?	É o processo em que organismos fotossintéticos usam energia da luz para produzir compostos orgânicos a partir de dióxido de carbono e água; em muitos casos, há liberação de oxigênio.
Qual é a diferença entre RAM e armazenamento?	RAM é memória de trabalho volátil usada durante a execução; armazenamento, como SSD ou HD, mantém dados mesmo quando o dispositivo é desligado.
O que é uma CPU?	É a unidade central de processamento responsável por executar instruções e realizar operações computacionais.
O que é uma função em programação?	É um bloco de código reutilizável que executa uma tarefa e pode receber argumentos e retornar um resultado.
O que é uma variável?	É um nome associado a um valor que pode ser usado e, conforme a linguagem, atualizado durante a execução.
O que é Git?	Git é um sistema distribuído de controle de versões.
Git e GitHub são a mesma coisa?	Não. Git é o sistema de controle de versões; GitHub é uma plataforma que hospeda repositórios Git e oferece recursos de colaboração.
O que é uma branch?	É uma linha de desenvolvimento separada que permite trabalhar em mudanças sem alterar diretamente outra linha.
O que é um commit?	É um registro de alterações no histórico de um repositório Git.
O que é uma API?	É uma interface que define como componentes de software podem se comunicar.
O que é um endpoint?	É um ponto específico de uma API que expõe uma operação ou recurso.'''


def load_base() -> list[dict]:
    items = []
    if not BASE_FILE.exists():
        return items
    with BASE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("user") and row.get("assistant"):
                items.append({"user": row["user"].strip(), "assistant": row["assistant"].strip()})
    return items


def main() -> None:
    rng = random.Random(SEED)
    base = load_base()
    accepted: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for item in base:
        k = (item["user"], item["assistant"])
        if k not in seen:
            seen.add(k)
            accepted.append(item)

    seeds = []
    for raw in SEED_TEXT.splitlines():
        if not raw.strip() or "\t" not in raw:
            continue
        user, assistant = raw.split("\t", 1)
        seeds.append((user.strip(), assistant.strip()))

    rng.shuffle(seeds)
    candidates = []
    for user, assistant in seeds:
        variants = VARIANTS[:]
        rng.shuffle(variants)
        for template in variants:
            prompt = template.format(user)
            item = {"user": prompt, "assistant": assistant}
            if (item["user"], item["assistant"]) not in seen:
                candidates.append(item)
    rng.shuffle(candidates)

    for item in candidates:
        if len(accepted) >= TARGET:
            break
        k = (item["user"], item["assistant"])
        if k in seen:
            continue
        seen.add(k)
        accepted.append(item)

    if len(accepted) < 500:
        raise RuntimeError(f"Dataset gerado com {len(accepted)} exemplos; mínimo exigido: 500")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="\n") as f:
        for item in accepted:
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"Base v0.6: {len(base)}")
    print(f"Total v0.7: {len(accepted)}")
    print(f"Saída: {OUTPUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
