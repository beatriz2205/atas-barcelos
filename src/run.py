"""3) Execução: para cada pergunta recupera os excertos UMA vez e pede a resposta
a cada modelo configurado.

Resultado: outputs/<modelo>/<pipeline>/answers.jsonl + meta.json (rastreabilidade
da execução: parâmetros efetivos, data, commit do Git).

Uso:
  python src/run.py                      # todos os modelos, todas as perguntas
  python src/run.py --pipeline baseline
  python src/run.py --models qwen3-8b    # só alguns modelos
  python src/run.py --limit 5            # teste rápido
"""
import argparse
import datetime
import json
import subprocess
from pathlib import Path

from tqdm import tqdm

from common import load_config, read_jsonl, resolve_pipeline, write_jsonl
from rag import Generator, Retriever


def meta_da_execucao(cfg: dict, pipeline_name: str, pipeline_cfg: dict, modelo: dict,
                     perguntas: list[dict], data_hora=None) -> dict:
    """Parâmetros efetivos e contexto de uma execução, para rastreabilidade completa."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        commit = None
    return {
        "data_hora": data_hora or datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": commit,
        # servidor e token_env registados; a chave só existe no .env, nunca aqui
        "ollama_host": cfg["ollama"]["host"],
        "modelo": modelo,
        "pipeline": {"nome": pipeline_name, **pipeline_cfg},
        "n_perguntas": len(perguntas),
        "ids_perguntas": [q["id"] for q in perguntas],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--pipeline", help="Nome do pipeline definido em config.yaml")
    args = ap.parse_args()

    cfg = load_config()
    pipeline_name, pipeline_cfg, cfg = resolve_pipeline(cfg, args.pipeline)
    perguntas = [q for q in read_jsonl(cfg["paths"]["questions"])
                 if "PREENCHER" not in str(q)][: args.limit]  # ignora modelos por preencher
    modelos = [m for m in cfg["models"] if not args.models or m["name"] in args.models]

    # recuperação fixa -> todos os modelos recebem exatamente o mesmo contexto
    retriever = Retriever(cfg, pipeline_name)
    recuperado = {q["id"]: retriever(q["pergunta"]) for q in tqdm(perguntas, desc="retrieval")}

    for m in modelos:
        gen = Generator(m, cfg)
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
                "pipeline": pipeline_name,
                **r,
                "contextos": [c["text"] for c in ctx],
                # data da reunião identifica a ata (os n.º repetem-se entre mandatos)
                "contextos_datas": [c["metadata"].get("data_reuniao") for c in ctx],
            })
        out_dir = Path(cfg["paths"]["outputs_dir"]) / m["name"] / pipeline_name
        answer_path = out_dir / "answers.jsonl"
        write_jsonl(answer_path, linhas)
        # rastreabilidade: o meta.json guarda os parâmetros efetivos, a data e o commit,
        # para que qualquer resultado seja reproduzível e auditável depois
        meta_path = out_dir / "meta.json"
        meta_path.write_text(json.dumps(
            meta_da_execucao(cfg, pipeline_name, pipeline_cfg, m, perguntas),
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {answer_path}")


if __name__ == "__main__":
    main()
