"""Utilitários partilhados: config, leitura/escrita JSONL e cliente de LLM."""
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # tornar caminhos absolutos a partir da raiz do projeto
    cfg["paths"] = {k: str(ROOT / v) for k, v in cfg["paths"].items()}
    return cfg


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip() and not l.lstrip().startswith("//")]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_client(model_cfg: dict):
    """Cliente OpenAI-compatível para qualquer servidor (vLLM, Ollama, ...)."""
    from openai import OpenAI

    key_env = model_cfg.get("api_key_env")
    api_key = os.environ.get(key_env, "") if key_env else "not-needed"
    return OpenAI(base_url=model_cfg["base_url"], api_key=api_key or "not-needed")
