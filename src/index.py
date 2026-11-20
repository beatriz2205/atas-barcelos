"""2) Indexação: chunks -> embeddings -> ChromaDB (data/pipelines/<pipeline>/chroma).

Os embeddings são calculados pelo `Embedder` (common.py), segundo a secção `embeddings`
do pipeline: no servidor Ollama (embeddinggemma) ou localmente (bge-m3).

Uso:  python src/index.py [--pipeline baseline]
"""
import argparse

import chromadb

from common import Embedder, load_config, pipeline_paths, read_jsonl, resolve_pipeline

COLLECTION = "atas"


def get_collection(cfg, pipeline_name=None, reset=False):
    """Coleção do pipeline. Os vetores são sempre fornecidos pelo nosso Embedder, por isso
    a coleção não tem função de embeddings própria; o modelo usado fica nos metadados."""
    name, _, resolved_cfg = resolve_pipeline(cfg, pipeline_name)
    client = chromadb.PersistentClient(path=pipeline_paths(cfg, name)["chroma_dir"])
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    # O índice é sempre só semântico; o BM25 da pesquisa híbrida é calculado no Retriever.
    return client.get_or_create_collection(
        COLLECTION,
        embedding_function=None,
        metadata={"hnsw:space": "cosine", "embedding_model": resolved_cfg["embeddings"]["model"]},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", help="Nome do pipeline definido em config.yaml")
    args = ap.parse_args()
    cfg = load_config()
    pipeline_name, _, rcfg = resolve_pipeline(cfg, args.pipeline)
    chunks = read_jsonl(pipeline_paths(cfg, pipeline_name)["chunks"])
    embedder = Embedder(rcfg)
    col = get_collection(cfg, pipeline_name, reset=True)
    print(f"[{pipeline_name}] embeddings com {embedder.model} ({embedder.provider})")
    B = 64
    for i in range(0, len(chunks), B):
        lote = chunks[i:i + B]
        col.add(
            ids=[c["id"] for c in lote],
            documents=[c["text"] for c in lote],
            embeddings=embedder.embed_documents([c["text"] for c in lote]),
            # Chroma não aceita None nos metadados
            metadatas=[{k: ("" if v is None else v) for k, v in c["metadata"].items()} for c in lote],
        )
        print(f"{min(i + B, len(chunks))}/{len(chunks)} indexados", flush=True)


if __name__ == "__main__":
    main()
