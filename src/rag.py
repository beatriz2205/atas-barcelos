"""Componentes RAG: recuperação (comum a todos os modelos) e geração (por modelo)."""
import re
import time
import unicodedata

from common import Embedder, make_client, ollama_chat, resolve_pipeline
from index import get_collection

NAO_CONSTA = "A informação não consta das atas fornecidas."

def build_system_prompt(prompt_cfg: dict) -> str:
    regras = [
        "Responde APENAS com base nos excertos fornecidos. Não uses conhecimento externo."
        if prompt_cfg["strict_grounding"] else "Usa os excertos como fonte principal.",
        f'Se a resposta não estiver nos excertos, responde exatamente: "{prompt_cfg["abstention_text"]}"',
        f'Responde em {prompt_cfg["language"]}, de forma concisa e factual.',
    ]
    if prompt_cfg["require_citations"]:
        regras.append("Indica no fim a(s) fonte(s) no formato (Ata n.º X, AAAA-MM-DD).")
    return "És um assistente que responde a perguntas sobre as atas das reuniões da Câmara Municipal de Barcelos.\nRegras:\n" + "\n".join(
        f"{i}. {regra}" for i, regra in enumerate(regras, 1)
    )


# Palavras demasiado comuns para distinguirem excertos (como no BM25 do Qdrant/FastEmbed).
STOPWORDS_PT = set("""a ao aos as à às com como da das de do dos e é em entre era essa esse
esta este eu foi foram há isso isto já lhe mais mas me mesmo muito na nas não nem no nos
o os ou para pela pelas pelo pelos por qual quando que quem se sem ser seu sua são também
te tem tinha um uma umas uns foi ser ter""".split())


def _sem_acentos(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def tokenizar(texto: str, lingua: str | None = "portuguese") -> list[str]:
    """Tokens para BM25. Com `lingua`, retira palavras comuns e reduz cada palavra à raiz
    (stemming Snowball: "aprovada", "aprovação" e "aprovar" -> "aprov"). Sem `lingua`,
    apenas minúsculas e sem acentos."""
    palavras = re.findall(r"\w+", texto.lower())
    if lingua:
        import snowballstemmer

        palavras = [w for w in palavras if w not in STOPWORDS_PT]
        palavras = snowballstemmer.stemmer(lingua).stemWords(palavras)
    return [_sem_acentos(w) for w in palavras]


def _minmax(valores: dict) -> dict:
    lo, hi = min(valores.values()), max(valores.values())
    return {k: (v - lo) / (hi - lo) if hi > lo else 0.0 for k, v in valores.items()}


def _rrf(rankings: list[list[str]], k: int = 60) -> dict:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (k + posição de d), posições a partir de 1.
    Funde pela POSIÇÃO, porque os scores do cosseno e do BM25 não são comparáveis."""
    score = {}
    for lista in rankings:
        for pos, i in enumerate(lista, 1):
            score[i] = score.get(i, 0.0) + 1.0 / (k + pos)
    return score


class Retriever:
    """Recuperação partilhada por todos os modelos. Opções (secção `retrieval` do pipeline):
      top_k:        n.º de excertos devolvidos;
      hybrid:       pesquisa híbrida semântica + BM25, com
                      fusion: rrf (predefinição) | weighted,
                      rrf_k: 60, weights: {semantic, sparse} (só em weighted),
                      bm25_language: portuguese (stemming + palavras comuns) | null;
      date_filter:  se a pergunta indicar a data de uma reunião existente, pesquisa só
                    nos excertos dessa ata (a pesquisa vetorial não serve para valores exatos);
      rerank:       {model, candidates}: recupera `candidates` excertos e um cross-encoder
                    (mais lento, mas lê pergunta e excerto em conjunto) reordena-os e fica com top_k.
    `hybrid_weights` (formato antigo) equivale a hybrid com fusion: weighted.
    """

    def __init__(self, cfg, pipeline_name=None):
        self.col = get_collection(cfg, pipeline_name)
        _, _, resolved_cfg = resolve_pipeline(cfg, pipeline_name)
        ret = resolved_cfg["retrieval"]
        self.k = ret["top_k"]
        # Regra de ouro: pesquisar com o mesmo modelo de embeddings com que se indexou.
        self.embedder = Embedder(resolved_cfg)
        indexado = (self.col.metadata or {}).get("embedding_model")
        if indexado and indexado != self.embedder.model:
            raise RuntimeError(f"O índice foi feito com {indexado}, mas o pipeline usa "
                               f"{self.embedder.model}. Corre de novo o index.py.")
        if self.col.count() == 0:
            raise RuntimeError("O índice está vazio. Corre primeiro o ingest.py e o index.py.")

        self.hybrid = ret.get("hybrid")
        if self.hybrid is None and ret.get("hybrid_weights"):
            self.hybrid = {"fusion": "weighted", "weights": ret["hybrid_weights"]}
        self.date_filter = ret.get("date_filter", False)
        self.rerank = ret.get("rerank")
        if self.rerank:
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(self.rerank["model"])

        todos = self.col.get(include=["documents", "metadatas"])
        self.docs = {i: (d, m) for i, d, m in zip(todos["ids"], todos["documents"], todos["metadatas"])}
        self.ids = list(self.docs)
        self.datas = {m.get("data_reuniao") for _, m in self.docs.values()}
        if self.hybrid:
            # O Chroma local não tem pesquisa lexical: o BM25 é calculado aqui,
            # sobre todos os chunks do pipeline (alguns milhares, cabe em memória).
            from rank_bm25 import BM25Okapi

            self.lingua = self.hybrid.get("bm25_language", "portuguese")
            self.bm25 = BM25Okapi([tokenizar(self.docs[i][0], self.lingua) for i in self.ids])

    def data_da_pergunta(self, pergunta: str) -> str | None:
        """Data (AAAA-MM-DD) de uma reunião do corpus mencionada na pergunta, ou None.
        Datas que não correspondem a nenhuma reunião (ex.: premissas falsas) são ignoradas."""
        from ingest import extrair_data

        data = extrair_data(pergunta)
        return data if data in self.datas else None

    def __call__(self, pergunta: str, k: int | None = None) -> list[dict]:
        k = k or self.k
        data = self.data_da_pergunta(pergunta) if self.date_filter else None
        n = max(k, self.rerank.get("candidates", 20)) if self.rerank else k
        if self.hybrid:
            res = self._hibrido(pergunta, n, data)
        else:
            res = self._semantico(pergunta, n, data)
        if self.rerank:
            scores = self.reranker.predict([(pergunta, c["text"]) for c in res])
            for c, s in zip(res, scores):
                c["score_rerank"] = round(float(s), 4)
            res = sorted(res, key=lambda c: c["score_rerank"], reverse=True)
        return res[:k]

    def _semantico(self, pergunta: str, k: int, data: str | None = None) -> list[dict]:
        args = {"query_embeddings": [self.embedder.embed_query(pergunta)], "n_results": k}
        if data:
            args["where"] = {"data_reuniao": data}
        r = self.col.query(**args)
        return [
            {"id": i, "text": d, "metadata": m, "distance": dist,
             "similaridade": round(1 - dist, 4), "filtro_data": data}
            for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])
        ]

    def _hibrido(self, pergunta: str, k: int, data: str | None = None) -> list[dict]:
        """Pesquisa semântica e BM25 sobre os mesmos candidatos (todos, ou só os da data
        filtrada), fundidas por RRF ou por soma ponderada dos scores normalizados."""
        candidatos = [i for i in self.ids if not data or self.docs[i][1].get("data_reuniao") == data]
        args = {"query_embeddings": [self.embedder.embed_query(pergunta)],
                "n_results": len(candidatos), "include": ["distances"]}
        if data:
            args["where"] = {"data_reuniao": data}
        r = self.col.query(**args)
        sem = {i: 1 - d for i, d in zip(r["ids"][0], r["distances"][0])}
        todos_bm25 = dict(zip(self.ids, self.bm25.get_scores(tokenizar(pergunta, self.lingua))))
        lex = {i: float(todos_bm25[i]) for i in candidatos}

        if self.hybrid.get("fusion", "rrf") == "rrf":
            rank_sem = sorted(sem, key=sem.get, reverse=True)
            # no BM25 só contam os excertos com alguma palavra da pergunta
            rank_lex = [i for i in sorted(lex, key=lex.get, reverse=True) if lex[i] > 0]
            score = _rrf([rank_sem, rank_lex], self.hybrid.get("rrf_k", 60))
        else:
            w = self.hybrid["weights"]
            sem_n, lex_n = _minmax(sem), _minmax(lex)
            score = {i: w["semantic"] * sem_n.get(i, 0.0) + w["sparse"] * lex_n[i] for i in candidatos}

        top = sorted(score, key=score.get, reverse=True)[:k]
        return [
            {"id": i, "text": self.docs[i][0], "metadata": self.docs[i][1],
             "score": round(score[i], 6), "similaridade": round(sem.get(i, 0.0), 4),
             "score_bm25": round(lex.get(i, 0.0), 4), "filtro_data": data}
            for i in top
        ]


def build_prompt(pergunta: str, contextos: list[str]) -> str:
    blocos = "\n\n".join(f"### Excerto {i + 1}\n{c}" for i, c in enumerate(contextos))
    return f"Excertos das atas:\n\n{blocos}\n\n---\nPergunta: {pergunta}\nResposta:"


class Generator:
    def __init__(self, model_cfg: dict, rag_cfg: dict):
        self.cfg = model_cfg
        self.gen = rag_cfg["generation"]
        self.system_prompt = build_system_prompt(rag_cfg["prompt"])
        self.ollama_cfg = rag_cfg.get("ollama")
        self.client = make_client(model_cfg, self.ollama_cfg)

    def __call__(self, pergunta: str, contextos: list[str]) -> dict:
        t0 = time.perf_counter()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": build_prompt(pergunta, contextos)},
        ]
        if self.ollama_cfg:
            gen = {**self.gen, **({"stop": self.cfg["stop"]} if self.cfg.get("stop") else {})}
            r = ollama_chat(self.client, self.cfg["model"], messages, gen, self.ollama_cfg)
            resposta = (r.message.content or "").strip()
            tokens_in = getattr(r, "prompt_eval_count", None)
            tokens_out = getattr(r, "eval_count", None)
        else:
            r = self.client.chat.completions.create(
                model=self.cfg["model"], messages=messages,
                temperature=self.gen["temperature"], max_tokens=self.gen["max_tokens"],
            )
            u = getattr(r, "usage", None)
            resposta = (r.choices[0].message.content or "").strip()
            tokens_in = getattr(u, "prompt_tokens", None)
            tokens_out = getattr(u, "completion_tokens", None)
        return {
            "resposta": resposta,
            "latencia_s": round(time.perf_counter() - t0, 3),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
