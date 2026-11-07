# Avaliação de LLMs com RAG sobre as atas da Câmara Municipal de Barcelos

Objetivo: comparar vários LLMs a extrair informação das atas das reuniões de câmara,
usando o **mesmo pipeline RAG** para todos e avaliando com **RAGAS** + métricas próprias.

```
PDFs atas ──► ingest.py ──► chunks.jsonl ──► index.py ──► ChromaDB (bge-m3)
                                                               │  top-k (igual p/ todos)
questions.jsonl ──────────────────────────────► run.py ◄───────┘
                                                  │  modelo A, B, C... (API OpenAI-compatível)
                                                  ▼
                                          answers_<modelo>.jsonl ──► evaluate.py ──► CSVs
```

## 1. Decisões de desenho (para justificar no relatório)

| Decisão | Porquê |
|---|---|
| Recuperação feita **uma vez** e partilhada por todos os modelos | Isola a variável em estudo: só o gerador muda. Comparação justa. |
| Chunk por **"PROPOSTA N.º X"** + janela de 1500 caracteres | Cada deliberação fica num chunk coerente; evita misturar propostas. |
| Cabeçalho `[Ata n.º X \| ordinária \| AAAA-MM-DD]` em cada chunk | As atas escrevem datas por extenso; o cabeçalho torna perguntas por data/n.º recuperáveis. |
| `temperature = 0` | Respostas reprodutíveis. |
| Prompt obriga a responder "não consta" quando não há informação | Permite medir alucinação de forma objetiva. |
| Juiz RAGAS ≠ modelos avaliados | Evita que o juiz favoreça o próprio modelo. |
| Embeddings multilingues (bge-m3) | As atas estão em português europeu. |

## 2. Que modelos comparar

Juntar modelos que variem **num eixo de cada vez**, para as conclusões serem interpretáveis:

- **Mesma gama de tamanho (7–9B), famílias diferentes** — ex.: Llama 3.x 8B, Qwen 2.5 7B, Mistral 7B, Gemma 9B. Pergunta: que família lida melhor com PT-PT administrativo?
- **Mesma família, tamanhos diferentes** — ex.: Qwen 7B vs 14B vs 32B. Pergunta: compensa o custo de um modelo maior?
- **Genérico vs. treinado para português** — se o professor tiver algum modelo PT-PT (ex.: Gervásio, AMALIA), comparar com um genérico do mesmo tamanho.
- (Opcional) **Um modelo comercial como teto de referência** (GPT/Claude/Gemini).

Assim que souberes a lista do professor, encaixa-a nestes eixos e põe-na no `config.yaml`.

## 3. Conjunto de perguntas (`data/questions.jsonl`)

8 tipos, para ver *onde* cada modelo falha e não só *quanto*:

| Tipo | Exemplo | O que testa |
|---|---|---|
| `factual_simples` | Data da reunião da ata n.º 24 | Extração direta |
| `votacao` | Foi aprovada por unanimidade? Quem votou contra? | Ler resultados de votação |
| `numerico` | Valor do apoio à associação X | Precisão em números/euros |
| `entidades` | Com que entidades foi assinado o protocolo? | Nomes de pessoas/entidades |
| `temporal` | Quando foi aprovado pela 1.ª vez X? | Raciocínio sobre datas |
| `multi_ata` | Em quantas reuniões se discutiu X? | Juntar informação de várias atas |
| `sintese` | Resume as deliberações da reunião X | Resumo fiel |
| `sem_resposta` | Orçamento do "aeroporto de Barcelos" | **Alucinação** — deve recusar |

Campos: `pergunta`, `resposta_referencia` (escrita por ti a partir da ata), `valores_esperados`
(números/nomes que têm de aparecer na resposta), `ata_fonte` (n.º das atas onde está a resposta).
Linhas com `PREENCHER` são ignoradas até serem preenchidas. **Meta: 40–60 perguntas, ≥5 por tipo**,
idealmente validadas por 2 pessoas do grupo.

## 4. Métricas

**RAGAS (com LLM-juiz)**

| Métrica | Mede | Componente |
|---|---|---|
| `faithfulness` | A resposta é suportada pelos excertos? (anti-alucinação) | Geração |
| `answer_relevancy` | A resposta responde mesmo à pergunta? | Geração |
| `factual_correctness` | Afirmações da resposta vs. referência (F1) | Geração |
| `llm_context_precision_with_reference` | Os excertos úteis aparecem no topo? | Recuperação |
| `context_recall` | Os excertos cobrem a referência? | Recuperação |

**Próprias (determinísticas — não dependem do juiz, servem para validar o RAGAS)**

| Métrica | Definição |
|---|---|
| `valores_corretos` | % de `valores_esperados` presentes na resposta |
| `abstencao_correta` | Respondeu quando havia resposta / recusou quando não havia |
| `alucinacao` | Respondeu a uma pergunta `sem_resposta` |
| `falsa_recusa` | Disse "não consta" quando a resposta estava lá |
| `retrieval_hit@k` | Alguma das atas certas está nos top-k excertos |
| `latencia_s`, `tokens_out` | Custo/velocidade |

Como a recuperação é igual para todos, as métricas de contexto vão dar ≈ igual entre modelos — servem
para mostrar que os erros vêm da geração e não da pesquisa (ou para comparar configurações de retrieval).

## 5. Como correr

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. pôr os PDFs das atas em data/atas/  (cm-barcelos.pt → Reuniões de Câmara)
python src/ingest.py          # verifica no output se n.º de ata e datas foram bem extraídos
python src/index.py

# 2. para testar já, sem os modelos do professor: Ollama local
ollama pull qwen2.5:7b && ollama pull llama3.1:8b && ollama pull qwen2.5:14b

python src/run.py --limit 3                  # teste rápido
python src/run.py                            # tudo
python src/evaluate.py --sem-ragas           # métricas próprias (rápido)
python src/evaluate.py                       # + RAGAS (lento: várias chamadas ao juiz/pergunta)
```

Resultados em `outputs/`: `resumo_por_modelo.csv`, `resumo_por_tipo.csv`, `resultados_por_pergunta.csv`.

## 6. Próximos passos

1. Escolher **10–20 atas** (ex.: um ano completo, para permitir perguntas `multi_ata`) e descarregar.
2. Correr `ingest.py` e confirmar a extração. Se algum PDF for digitalizado (sem texto), precisa de OCR.
3. Ler as atas e preencher as perguntas — é a parte mais demorada e a mais importante.
4. Testar o pipeline com Ollama para ter resultados preliminares.
5. Quando houver acesso: pôr os modelos do professor no `config.yaml` e correr tudo.
6. Análise: tabela por modelo, gráfico por tipo de pergunta, exemplos de erros típicos, e
   correlação entre `faithfulness` (RAGAS) e `valores_corretos` (determinística).
