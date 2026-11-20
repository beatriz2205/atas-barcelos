"""Verifica a recuperação SEM chamar nenhum LLM: para cada pergunta com `ata_fonte`,
as atas certas (pela data da reunião) aparecem nos top-k excertos?

Serve para separar erros de pesquisa de erros de geração: se a ata certa não é
recuperada, nenhum modelo pode acertar. Também permite comparar pipelines
(ex.: baseline vs hybrid_baseline) antes de gastar chamadas aos modelos.

Uso:
  python src/check_retrieval.py                         # pipeline predefinido; k = 3, top_k e 10
  python src/check_retrieval.py --pipeline hybrid_baseline
  python src/check_retrieval.py --k 5 20                # outros valores de k
"""
import argparse
from collections import defaultdict

from common import load_config, read_jsonl, resolve_pipeline
from rag import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", help="Nome do pipeline definido em config.yaml")
    ap.add_argument("--k", type=int, nargs="*")
    args = ap.parse_args()

    cfg = load_config()
    pipeline_name, _, rcfg = resolve_pipeline(cfg, args.pipeline)
    k0 = rcfg["retrieval"]["top_k"]
    ks = args.k or sorted({3, k0, 10})
    kmax = max(ks)

    perguntas = [q for q in read_jsonl(cfg["paths"]["questions"])
                 if q.get("ata_fonte") and "PREENCHER" not in str(q)]
    retriever = Retriever(cfg, pipeline_name)

    por_tipo = defaultdict(lambda: defaultdict(list))
    falhas = []
    for q in perguntas:
        datas = [c["metadata"].get("data_reuniao") for c in retriever(q["pergunta"], k=kmax)]
        fontes = set(q["ata_fonte"])
        for k in ks:
            top = set(datas[:k])
            hit = float(bool(fontes & top))                 # alguma ata certa
            recall = len(fontes & top) / len(fontes)        # fração das atas certas (multi_ata)
            for grupo in (q["tipo"], "TOTAL"):
                por_tipo[grupo][f"hit@{k}"].append(hit)
                por_tipo[grupo][f"recall@{k}"].append(recall)
        if not fontes & set(datas[:k0]):
            # posição (1-based) da 1.ª ata certa nos top-kmax, se aparecer
            pos = next((i + 1 for i, d in enumerate(datas) if d in fontes), None)
            falhas.append((q["id"], q["tipo"], sorted(fontes), datas[:k0], pos))

    print(f"Pipeline: {pipeline_name}\n")
    cols = [f"{m}@{k}" for k in ks for m in ("hit", "recall")]
    print(f"{'tipo':<36}{'n':>3}" + "".join(f"{c:>11}" for c in cols))
    for tipo in sorted(por_tipo, key=lambda t: (t == "TOTAL", t)):
        v = por_tipo[tipo]
        n = len(v[cols[0]])
        print(f"{tipo:<36}{n:>3}" + "".join(f"{sum(v[c]) / n:>11.2f}" for c in cols))

    print(f"\nFalhas em top-{k0} ({len(falhas)}/{len(perguntas)}):")
    for qid, tipo, fontes, top, pos in falhas:
        onde = f"aparece em #{pos}" if pos else f"fora dos top-{kmax}"
        print(f"  {qid} [{tipo}] certa: {fontes} | recuperadas: {top} | {onde}")


if __name__ == "__main__":
    main()
