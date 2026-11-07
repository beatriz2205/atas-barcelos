"""2) Indexação: chunks -> embeddings -> ChromaDB (data/chroma).

Uso:  python src/index.py
"""
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from common import load_config, read_jsonl

COLLECTION = "atas"


def get_collection(cfg, reset=False):
    client = chromadb.PersistentClient(path=cfg["paths"]["chroma_dir"])
    emb = SentenceTransformerEmbeddingFunction(model_name=cfg["embeddings"]["model"])
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(COLLECTION, embedding_function=emb,
                                           metadata={"hnsw:space": "cosine"})


def main():
    cfg = load_config()
    chunks = read_jsonl(cfg["paths"]["chunks"])
    col = get_collection(cfg, reset=True)
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
