"""Componentes RAG: recuperação (comum a todos os modelos) e geração (por modelo)."""
import time

from common import make_client
from index import get_collection

NAO_CONSTA = "A informação não consta das atas fornecidas."

SYSTEM_PROMPT = f"""És um assistente que responde a perguntas sobre as atas das reuniões da Câmara Municipal de Barcelos.
Regras:
1. Responde APENAS com base nos excertos fornecidos. Não uses conhecimento externo.
2. Se a resposta não estiver nos excertos, responde exatamente: "{NAO_CONSTA}"
3. Responde em português europeu, de forma concisa e factual.
4. Indica no fim a(s) fonte(s) no formato (Ata n.º X, AAAA-MM-DD)."""


class Retriever:
    def __init__(self, cfg):
        self.col = get_collection(cfg)
        self.k = cfg["retrieval"]["top_k"]

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
    def __init__(self, model_cfg: dict, gen_cfg: dict):
        self.cfg = model_cfg
        self.gen = gen_cfg
        self.client = make_client(model_cfg)

    def __call__(self, pergunta: str, contextos: list[str]) -> dict:
        t0 = time.perf_counter()
        r = self.client.chat.completions.create(
            model=self.cfg["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(pergunta, contextos)},
            ],
            temperature=self.gen["temperature"],
            max_tokens=self.gen["max_tokens"],
        )
        u = getattr(r, "usage", None)
        return {
            "resposta": (r.choices[0].message.content or "").strip(),
            "latencia_s": round(time.perf_counter() - t0, 3),
            "tokens_in": getattr(u, "prompt_tokens", None),
            "tokens_out": getattr(u, "completion_tokens", None),
        }
