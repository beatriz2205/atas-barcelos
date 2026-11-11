"""Componentes RAG: recuperação (comum a todos os modelos) e geração (por modelo)."""
import time

from common import make_client, ollama_chat, resolve_pipeline
from index import get_collection

NAO_CONSTA = "A informação não consta das atas fornecidas."

def build_system_prompt(prompt_cfg: dict) -> str:
    regras = [
        "Responde APENAS com base nos excertos fornecidos. Não uses conhecimento externo."
        if prompt_cfg["strict_grounding"] else "Usa os excertos como fonte principal.",
        f'Se a resposta não estiver nos excertos, responde exatamente: "{prompt_cfg["abstention_text"]}"',
        f'Responde em {prompt_cfg["language"]}, de forma concisa e factual.',
    ]
    if prompt_cfg["require_citations"]:
        regras.append("Indica no fim a(s) fonte(s) no formato (Ata n.º X, AAAA-MM-DD).")
    return "És um assistente que responde a perguntas sobre as atas das reuniões da Câmara Municipal de Barcelos.\nRegras:\n" + "\n".join(
        f"{i}. {regra}" for i, regra in enumerate(regras, 1)
    )


class Retriever:
    def __init__(self, cfg, pipeline_name=None):
        self.col = get_collection(cfg, pipeline_name)
        _, _, resolved_cfg = resolve_pipeline(cfg, pipeline_name)
        self.k = resolved_cfg["retrieval"]["top_k"]

    def __call__(self, pergunta: str, k: int | None = None) -> list[dict]:
        r = self.col.query(query_texts=[pergunta], n_results=k or self.k)
        return [
            {"id": i, "text": d, "metadata": m, "distance": dist}
            for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])
        ]


def build_prompt(pergunta: str, contextos: list[str]) -> str:
    blocos = "\n\n".join(f"### Excerto {i + 1}\n{c}" for i, c in enumerate(contextos))
    return f"Excertos das atas:\n\n{blocos}\n\n---\nPergunta: {pergunta}\nResposta:"


class Generator:
    def __init__(self, model_cfg: dict, rag_cfg: dict):
        self.cfg = model_cfg
        self.gen = rag_cfg["generation"]
        self.system_prompt = build_system_prompt(rag_cfg["prompt"])
        self.ollama_cfg = rag_cfg.get("ollama")
        self.client = make_client(model_cfg, self.ollama_cfg)

    def __call__(self, pergunta: str, contextos: list[str]) -> dict:
        t0 = time.perf_counter()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": build_prompt(pergunta, contextos)},
        ]
        if self.ollama_cfg:
            r = ollama_chat(self.client, self.cfg["model"], messages, self.gen, self.ollama_cfg)
            resposta = (r.message.content or "").strip()
            tokens_in = getattr(r, "prompt_eval_count", None)
            tokens_out = getattr(r, "eval_count", None)
        else:
            r = self.client.chat.completions.create(
                model=self.cfg["model"], messages=messages,
                temperature=self.gen["temperature"], max_tokens=self.gen["max_tokens"],
            )
            u = getattr(r, "usage", None)
            resposta = (r.choices[0].message.content or "").strip()
            tokens_in = getattr(u, "prompt_tokens", None)
            tokens_out = getattr(u, "completion_tokens", None)
        return {
            "resposta": resposta,
            "latencia_s": round(time.perf_counter() - t0, 3),
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        }
