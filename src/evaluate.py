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
import re
import unicodedata
from pathlib import Path

import pandas as pd

from common import load_config, read_jsonl, resolve_pipeline


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
    from langchain_huggingface import HuggingFaceEmbeddings

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
        emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=pipeline_cfg["embeddings"]["model"])
        )
        ds = EvaluationDataset(samples=[
            SingleTurnSample(user_input=r.pergunta, response=r.resposta,
                             retrieved_contexts=list(r.contextos), reference=r.resposta_referencia)
            for r in sub.itertuples()
        ])
        res = evaluate(ds, metrics=metricas, llm=judge, embeddings=emb,
                       run_config=RunConfig(max_workers=4, timeout=180))
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
    linhas = [l for f in sorted(out.glob("answers_*.jsonl")) for l in read_jsonl(f)]
    if not linhas:
        raise SystemExit("Sem respostas em outputs/. Corre primeiro src/run.py")
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


if __name__ == "__main__":
    main()
