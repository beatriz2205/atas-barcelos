# Avaliação de LLMs com RAG sobre as atas da Câmara Municipal de Barcelos

Objetivo: comparar vários LLMs a extrair informação das atas das reuniões de câmara,
usando o **mesmo pipeline RAG** para todos e avaliando com **RAGAS** + métricas próprias.

```
PDFs atas ──► OCRmyPDF ──► ingest.py ──► chunks.jsonl ──► index.py ──► ChromaDB (bge-m3)
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

## 2. Modelos selecionados

Foram selecionados seis modelos na gama de **8–14B parâmetros**. Qwen 3 8B e
Gervásio 8B PT-PT cobrem, respetivamente, uma referência generalista e uma
especializada em português europeu. AMALIA 0626 SFT 9B e AMALIA 0626 DPO 9B formam
uma comparação controlada: mantêm família e escala, variando a estratégia de
afinação. Phi-4 14B e Ministral 3 14B acrescentam dois modelos ligeiramente
maiores, sem entrar na gama de dezenas ou centenas de milhares de milhões de
parâmetros. A diferença de escala é registada na análise: as comparações mais
controladas são entre AMALIA SFT/DPO, os modelos de 8–9B e os dois de 14B.

O orquestrador/juiz RAGAS é `gemma4:26b`: tem capacidade superior aos geradores,
mas é substancialmente mais económico do que um modelo de 119B. Apenas produz as
avaliações RAGAS, nunca respostas incluídas na comparação. A configuração efetiva
está em `config.yaml`.

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
(números/nomes que têm de aparecer na resposta), `ata_fonte` (datas AAAA-MM-DD das reuniões onde está a resposta — os n.º das atas repetem-se entre mandatos).
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

# Cada membro do grupo faz isto uma vez, no seu clone local.
cp .env.example .env
# editar .env e definir OLLAMA_TOKEN com a sua própria chave
python src/check_ollama.py  # confirma a chave e o acesso antes de correr a experiência

# 1. pôr os PDFs das atas em data/atas/  (cm-barcelos.pt → Reuniões de Câmara)
# O OCRmyPDF requer também Tesseract e o pacote de língua portuguesa no sistema.
# Ex. Debian/Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-por ghostscript
python src/ingest.py --pipeline baseline
python src/index.py --pipeline baseline

python src/run.py --pipeline baseline --limit 3  # teste rápido
python src/run.py --pipeline baseline            # tudo
python src/evaluate.py --sem-ragas           # métricas próprias (rápido)
python src/evaluate.py                       # + RAGAS (lento: várias chamadas ao juiz/pergunta)
```

O cliente usa o endpoint Ollama autenticado `afonsinho.di.uminho.pt:4590`. A chave
é lida da variável `OLLAMA_TOKEN` (ou automaticamente do ficheiro local `.env`),
nunca de `config.yaml`; `.env` está ignorado pelo Git. Cada pessoa pode assim correr
o projeto com a sua key, incluindo em simultâneo. Erros 504 são repetidos até três
vezes, com cinco segundos entre tentativas.

Resultados em `outputs/`: `resumo_por_modelo.csv`, `resumo_por_tipo.csv`, `resultados_por_pergunta.csv`.

### Pipelines RAG e comparação justa

Um pipeline é a configuração RAG inteira: OCR, limpeza/chunking, embeddings,
retrieval, geração e prompt. Todos os PDFs são processados pelo OCRmyPDF com
`--force-ocr`, inclusive quando já contêm texto; portanto, é sempre texto OCR que
chega aos embeddings e aos LLMs. Os PDFs OCR, chunks e índice ficam em
`data/pipelines/<pipeline>/`, impedindo reutilização acidental de artefactos.

O `baseline` reproduz as decisões RAG originais, com OCR adicionado. As variantes
incluídas alteram uma só decisão: `ocr_deskew`, `chunks_2500`, `top_k_10` e
`prompt_sem_fontes`. Para comparar modelos, usa sempre o mesmo pipeline. Para
comparar pipelines, fixa o mesmo modelo:

```bash
for p in baseline ocr_deskew chunks_2500 top_k_10 prompt_sem_fontes; do
  python src/ingest.py --pipeline "$p"
  python src/index.py --pipeline "$p"
  python src/run.py --pipeline "$p" --models qwen3-8b
done
python src/evaluate.py --sem-ragas
```

As respostas incluem o campo `pipeline`; os resumos são agrupados por `pipeline` e
`modelo`. Para recriar um OCR em cache, usa `python src/ingest.py --refazer-ocr`.

## 6. Próximos passos

1. Escolher **10–20 atas** (ex.: um ano completo, para permitir perguntas `multi_ata`) e descarregar.
2. Correr `ingest.py` e confirmar a extração. Se algum PDF for digitalizado (sem texto), precisa de OCR.
3. Ler as atas e preencher as perguntas — é a parte mais demorada e a mais importante.
4. Testar o pipeline com Ollama para ter resultados preliminares.
5. Quando houver acesso: pôr os modelos do professor no `config.yaml` e correr tudo.
6. Análise: tabela por modelo, gráfico por tipo de pergunta, exemplos de erros típicos, e
   correlação entre `faithfulness` (RAGAS) e `valores_corretos` (determinística).
