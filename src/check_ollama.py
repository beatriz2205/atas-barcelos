"""Verifica a chave individual e o acesso ao servidor Ollama.

Uso: python src/check_ollama.py [--model amalia-9b-0626-dpo:latest]
"""
import argparse
import time

from common import load_config, make_client, ollama_chat


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=cfg["models"][0]["model"],
                    help="Identificador Ollama a testar")
    args = ap.parse_args()

    client = make_client({"model": args.model}, cfg["ollama"])
    started = time.perf_counter()
    response = ollama_chat(
        client, args.model,
        [{"role": "user", "content": "Responde apenas: OK"}],
        {"temperature": 0.0, "max_tokens": 16}, cfg["ollama"],
    )
    print(f"Ligação autenticada a {args.model} em {time.perf_counter() - started:.2f}s")
    print((response.message.content or "").strip())


if __name__ == "__main__":
    main()
