"""2) Indexação: chunks -> embeddings -> ChromaDB (data/chroma).

Uso:  python src/index.py [--pipeline baseline]
"""
import argparse
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from common import load_config, pipeline_paths, read_jsonl, resolve_pipeline

COLLECTION = "atas"


def get_collection(cfg, pipeline_name=None, reset=False):
    name, pipeline, resolved_cfg = resolve_pipeline(cfg, pipeline_name)
    client = chromadb.PersistentClient(path=pipeline_paths(cfg, name)["chroma_dir"])
    emb = SentenceTransformerEmbeddingFunction(model_name=resolved_cfg["embeddings"]["model"])
    
    # Verifica se este pipeline usa hybrid search (semantic + sparse/BM25)
    is_hybrid = pipeline.get("retrieval", {}).get("hybrid_weights") is not None
    
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    
    # Para hybrid search, ativa o sparse index (BM25)
    collection_metadata = {"hnsw:space": "cosine"}
    if is_hybrid:
        collection_metadata["allow_sparse_index"] = True
    
    return client.get_or_create_collection(
        COLLECTION, 
        embedding_function=emb,
        metadata=collection_metadata
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", help="Nome do pipeline definido em config.yaml")
    args = ap.parse_args()
    cfg = load_config()
    pipeline_name, _, _ = resolve_pipeline(cfg, args.pipeline)
    chunks = read_jsonl(pipeline_paths(cfg, pipeline_name)["chunks"])
    col = get_collection(cfg, pipeline_name, reset=True)
    B = 64
    for i in range(0, len(chunks), B):
        lote = chunks[i:i + B]
        col.add(
            ids=[c["id"] for c in lote],
            documents=[c["text"] for c in lote],
            # Chroma não aceita None nos metadados
            metadatas=[{k: ("" if v is None else v) for k, v in c["metadata"].items()} for c in lote],
        )
        print(f"{min(i + B, len(chunks))}/{len(chunks)} indexados")


if __name__ == "__main__":
    main()
