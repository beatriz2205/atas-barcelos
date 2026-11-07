"""3) Execução: para cada pergunta recupera os excertos UMA vez e pede a resposta
a cada modelo configurado. Resultado: outputs/answers_<modelo>.jsonl

Uso:
  python src/run.py                      # todos os modelos, todas as perguntas
  python src/run.py --models qwen2.5-7b  # só alguns modelos
  python src/run.py --limit 5            # teste rápido
"""
import argparse
from pathlib import Path

from tqdm import tqdm

from common import load_config, read_jsonl, write_jsonl
from rag import Generator, Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    cfg = load_config()
    perguntas = [q for q in read_jsonl(cfg["paths"]["questions"])
                 if "PREENCHER" not in str(q)][: args.limit]  # ignora modelos por preencher
    modelos = [m for m in cfg["models"] if not args.models or m["name"] in args.models]

    # recuperação fixa -> todos os modelos recebem exatamente o mesmo contexto
    retriever = Retriever(cfg)
    recuperado = {q["id"]: retriever(q["pergunta"]) for q in tqdm(perguntas, desc="retrieval")}

    out_dir = Path(cfg["paths"]["outputs_dir"])
    for m in modelos:
        gen = Generator(m, cfg["generation"])
        linhas = []
        for q in tqdm(perguntas, desc=m["name"]):
            ctx = recuperado[q["id"]]
            try:
                r = gen(q["pergunta"], [c["text"] for c in ctx])
            except Exception as e:  # um modelo em baixo não deve parar o resto
                r = {"resposta": "", "erro": str(e), "latencia_s": None,
                     "tokens_in": None, "tokens_out": None}
            linhas.append({
                **q,
                "modelo": m["name"],
                **r,
                "contextos": [c["text"] for c in ctx],
                # data da reunião identifica a ata (os n.º repetem-se entre mandatos)
                "contextos_datas": [c["metadata"].get("data_reuniao") for c in ctx],
            })
        write_jsonl(out_dir / f"answers_{m['name']}.jsonl", linhas)
        print(f"-> {out_dir / f'answers_{m['name']}.jsonl'}")


if __name__ == "__main__":
    main()
