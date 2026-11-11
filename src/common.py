"""Utilitários partilhados: config, leitura/escrita JSONL e cliente de LLM."""
import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    # Cada membro mantém a sua própria chave em .env, ficheiro ignorado pelo Git.
    # Variáveis já exportadas no terminal têm precedência sobre esse ficheiro.
    load_dotenv(ROOT / ".env")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # tornar caminhos absolutos a partir da raiz do projeto
    cfg["paths"] = {k: str(ROOT / v) for k, v in cfg["paths"].items()}
    return cfg


def get_pipeline(cfg: dict, name: str | None = None) -> tuple[str, dict]:
    """Devolve o pipeline pedido (ou o definido como predefinido).

    Os artefactos derivados são isolados por pipeline. Assim, uma comparação
    entre, por exemplo, OCR básico e OCR com deskew usa exactamente o mesmo
    modelo, mas nunca reutiliza acidentalmente o índice do outro pipeline.
    """
    pipelines = cfg.get("pipelines", {})
    pipeline_name = name or cfg.get("default_pipeline")
    if not pipeline_name:
        raise ValueError("Falta 'default_pipeline' no config.yaml")
    if pipeline_name not in pipelines:
        disponiveis = ", ".join(sorted(pipelines)) or "nenhum"
        raise ValueError(f"Pipeline desconhecido: {pipeline_name}. Disponíveis: {disponiveis}")
    return pipeline_name, pipelines[pipeline_name]


PIPELINE_SECTIONS = ("ocr", "chunking", "embeddings", "retrieval", "generation", "prompt")


def resolve_pipeline(cfg: dict, name: str | None = None) -> tuple[str, dict, dict]:
    """Resolve um pipeline completo sem alterar a configuração original.

    Um pipeline é a condição experimental inteira, não apenas o OCR. As suas
    secções substituem as opções RAG efectivas usadas por ingestão, indexação e
    geração; manter o modelo fora desta estrutura permite fixá-lo ao comparar
    pipelines.
    """
    pipeline_name, pipeline = get_pipeline(cfg, name)
    missing = [section for section in PIPELINE_SECTIONS if section not in pipeline]
    if missing:
        raise ValueError(
            f"O pipeline '{pipeline_name}' não está completo: faltam {', '.join(missing)}."
        )
    resolved = deepcopy(cfg)
    for section in PIPELINE_SECTIONS:
        resolved[section] = deepcopy(pipeline[section])
    return pipeline_name, pipeline, resolved


def pipeline_paths(cfg: dict, pipeline_name: str) -> dict:
    """Caminhos exclusivos de um pipeline, derivados de paths.pipelines_dir."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", pipeline_name)
    base = Path(cfg["paths"]["pipelines_dir"]) / safe_name
    return {
        "base": str(base),
        "ocr_dir": str(base / "ocr"),
        "chunks": str(base / "chunks.jsonl"),
        "chroma_dir": str(base / "chroma"),
    }


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip() and not l.lstrip().startswith("//")]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_client(model_cfg: dict, ollama_cfg: dict | None = None):
    """Constrói o cliente configurado, sem expor a chave em ficheiros do projeto."""
    if ollama_cfg:
        from ollama import Client

        token_env = ollama_cfg["token_env"]
        token = os.environ.get(token_env)
        if not token:
            raise RuntimeError(
                f"Falta a variável {token_env}. Copia .env.example para .env e coloca a tua chave."
            )
        return Client(host=ollama_cfg["host"], headers={"Authorization": f"Bearer {token}"})

    from openai import OpenAI

    key_env = model_cfg.get("api_key_env")
    api_key = os.environ.get(key_env, "") if key_env else "not-needed"
    return OpenAI(base_url=model_cfg["base_url"], api_key=api_key or "not-needed")


def ollama_chat(client, model: str, messages: list[dict], generation: dict, ollama_cfg: dict):
    """Pedido chat Ollama com repetição apenas para timeouts transitórios (504)."""
    from ollama import ResponseError

    max_retries = ollama_cfg.get("max_retries", 3)
    delay = ollama_cfg.get("retry_delay_s", 5)
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat(
                model=model,
                messages=messages,
                options={"temperature": generation["temperature"],
                         "num_predict": generation["max_tokens"]},
            )
        except ResponseError as exc:
            if getattr(exc, "status_code", None) != 504 or attempt == max_retries:
                raise
            print(f"AVISO: timeout 504 em {model} ({attempt}/{max_retries}); nova tentativa em {delay}s.")
            time.sleep(delay)
