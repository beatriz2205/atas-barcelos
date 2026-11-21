"""4) Avaliação: métricas RAGAS (com LLM-juiz) + métricas determinísticas próprias.

Produz:
  outputs/resultados_por_pergunta.csv   (uma linha por modelo x pergunta)
  outputs/resumo_por_modelo.csv         (médias por modelo)
  outputs/resumo_por_tipo.csv           (médias por modelo x tipo de pergunta)

Uso:
  python src/evaluate.py                 # tudo
  python src/evaluate.py --sem-ragas     # só métricas determinísticas (rápido, sem juiz)
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from common import Embedder, load_config, read_jsonl, resolve_pipeline


# ---------------------------------------------------------------- métricas próprias
def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"(?<=\d)[.\s](?=\d{3}\b)", "", s)  # 12.500 / 12 500 -> 12500
    return re.sub(r"\s+", " ", s).strip()


RECUSA = ("nao consta", "nao e possivel", "nao ha informacao", "nao encontr",
          # pedidos de clarificação (pergunta ambígua) também contam como não responder
          "necessario indicar", "pode indicar", "indique", "especifique", "clarifica")


def absteve(resposta: str) -> bool:
    r = norm(resposta)
    return any(p in r for p in RECUSA)


def respondivel(row) -> bool:
    """A pergunta tem resposta nas atas? Decide-se pela referência, não pelo `tipo`:
    há tipos (ex.: resistencia_a_entidade_inexistente) cuja resposta certa é recusar."""
    return not absteve(row["resposta_referencia"])


def metricas_proprias(row) -> dict:
    resp = respondivel(row)
    abst = absteve(row["resposta"])
    out = {
        # 1 se o modelo fez a coisa certa: respondeu quando havia resposta / recusou quando não havia
        "abstencao_correta": float(abst != resp),
        "alucinacao": float(not resp and not abst),       # inventou resposta
        "falsa_recusa": float(resp and abst),              # recusou sem razão
    }
    esperados = row.get("valores_esperados") or []
    if resp and esperados:
        r = norm(row["resposta"])
        out["valores_corretos"] = sum(norm(v) in r for v in esperados) / len(esperados)
    fontes = row.get("ata_fonte") or []
    if fontes:  # métrica de retrieval: alguma ata certa (pela data da reunião) nos top-k?
        out["retrieval_hit@k"] = float(any(d in fontes for d in row["contextos_datas"]))
    return out


# ---------------------------------------------------------------- RAGAS
def avaliar_ragas(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    import os

    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, SingleTurnSample, evaluate
    from ragas.run_config import RunConfig
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, FactualCorrectness, LLMContextPrecisionWithReference,
                               LLMContextRecall, ResponseRelevancy)
    from langchain_core.embeddings import Embeddings

    class EmbedderLangchain(Embeddings):
        """Adapta o Embedder do projeto à interface do LangChain/RAGAS (ResponseRelevancy)."""

        def __init__(self, embedder):
            self.e = embedder

        def embed_documents(self, texts):
            # o ResponseRelevancy compara perguntas com perguntas: mesmo prefixo dos dois lados
            return self.e._embed([self.e.query_prefix + t for t in texts])

        def embed_query(self, text):
            return self.e.embed_query(text)

    j = cfg["judge"]
    if cfg.get("ollama"):
        from langchain_ollama import ChatOllama

        token_env = cfg["ollama"]["token_env"]
        token = os.environ.get(token_env)
        if not token:
            raise RuntimeError(f"Falta a variável {token_env}; configura a tua chave em .env.")
        judge_llm = ChatOllama(
            base_url=cfg["ollama"]["host"], model=j["model"], temperature=0,
            client_kwargs={"headers": {"Authorization": f"Bearer {token}"}},
        )
    else:
        key = os.environ.get(j["api_key_env"], "") if j.get("api_key_env") else "not-needed"
        judge_llm = ChatOpenAI(base_url=j["base_url"], model=j["model"],
                                api_key=key or "not-needed", temperature=0)
    judge = LangchainLLMWrapper(judge_llm)
    metricas = [
        Faithfulness(),                      # a resposta é suportada pelos excertos?
        ResponseRelevancy(),                 # a resposta responde à pergunta?
        FactualCorrectness(mode="f1"),       # resposta vs. referência (afirmações)
        LLMContextPrecisionWithReference(),  # excertos relevantes aparecem no topo?
        LLMContextRecall(),                  # os excertos cobrem a referência?
    ]

    resultados = []
    # perguntas sem resposta não fazem sentido para estas métricas -> só as respondíveis
    alvo = df[df.apply(respondivel, axis=1) & (df["resposta"].str.len() > 0)]
    for (pipeline, modelo), sub in alvo.groupby(["pipeline", "modelo"]):
        _, _, pipeline_cfg = resolve_pipeline(cfg, pipeline)
        emb = LangchainEmbeddingsWrapper(EmbedderLangchain(Embedder(pipeline_cfg)))
        ds = EvaluationDataset(samples=[
            SingleTurnSample(user_input=r.pergunta, response=r.resposta,
                             retrieved_contexts=list(r.contextos), reference=r.resposta_referencia)
            for r in sub.itertuples()
        ])
        res = evaluate(ds, metrics=metricas, llm=judge, embeddings=emb,
                       # timeout/menos workers: com 180 s e 4 em paralelo, um juiz de 26B
                       # num servidor partilhado ultrapassa o limite em >60% dos julgamentos
                       run_config=RunConfig(max_workers=2, timeout=600))
        rdf = res.to_pandas()
        rdf["id"], rdf["modelo"], rdf["pipeline"] = sub["id"].values, modelo, pipeline
        resultados.append(rdf.drop(columns=["user_input", "response", "retrieved_contexts",
                                            "reference"], errors="ignore"))
    return pd.concat(resultados) if resultados else pd.DataFrame(columns=["id", "modelo", "pipeline"])


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-ragas", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    out = Path(cfg["paths"]["outputs_dir"])
    # procura em outputs/<modelo>/<pipeline>/answers.jsonl (uma pasta de execução por
    # modelo x pipeline); arquivos em subpastas (ex.: variacoes/) e ficheiros com outros
    # nomes ficam de fora, para não entrar duas vezes no mesmo ensaio
    execucoes = sorted(p.parent for p in out.rglob("answers.jsonl")
                       if len(p.relative_to(out).parts) == 3)
    if not execucoes:
        raise SystemExit("Sem respostas em outputs/. Corre primeiro src/run.py")
    linhas = [l for d in execucoes for l in read_jsonl(d / "answers.jsonl")]
    df = pd.DataFrame(linhas)
    # Compatibilidade com resultados gerados antes da introdução de pipelines.
    if "pipeline" not in df:
        df["pipeline"] = cfg["default_pipeline"]
    df = pd.concat([df, pd.DataFrame([metricas_proprias(r) for r in linhas])], axis=1)

    if not args.sem_ragas:
        df = df.merge(avaliar_ragas(df, cfg), on=["id", "modelo", "pipeline"], how="left")

    cols_fora = ["contextos", "contextos_datas", "valores_esperados", "ata_fonte"]
    df.drop(columns=cols_fora, errors="ignore").to_csv(out / "resultados_por_pergunta.csv", index=False)

    num = [c for c in df.select_dtypes("number").columns if c not in ("tokens_in",)]
    resumo = df.groupby(["pipeline", "modelo"])[num].mean().round(3)
    resumo.to_csv(out / "resumo_por_modelo.csv")
    df.groupby(["pipeline", "modelo", "tipo"])[num].mean().round(3).to_csv(out / "resumo_por_tipo.csv")
    print(resumo.T.to_string())

    # cada pasta de execução fica autónoma: resultados da própria execução + contexto do juiz
    for d in execucoes:
        sub = df[(df["modelo"] == d.parts[-2]) & (df["pipeline"] == d.parts[-1])]
        if sub.empty:
            continue
        sub.drop(columns=cols_fora, errors="ignore").to_csv(d / "avaliacao.csv", index=False)
        sub.groupby("tipo")[num].mean().round(3).to_csv(d / "resumo_por_tipo.csv")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        meta["avaliacao"] = {
            "data_hora": pd.Timestamp.now().isoformat(timespec="seconds"),
            "ragas_ativado": not args.sem_ragas,
            "juiz": cfg.get("judge", {}).get("model"),
            "run_config_ragas": {"max_workers": 2, "timeout": 600} if not args.sem_ragas else None,
        }
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
