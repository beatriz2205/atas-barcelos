"""1) Ingestão: PDFs das atas -> texto limpo -> chunks com metadados (data/chunks.jsonl).

Particularidades das atas de Barcelos tratadas aqui:
  - linhas de tracejado "--------" usadas para preencher espaço;
  - datas escritas por extenso ("aos trinta dias do mês de janeiro de dois mil e dezassete");
  - capitular na 1.ª letra que o extrator perde ("cta n.º 24" em vez de "Acta n.º 24");
  - deliberações organizadas em "PROPOSTA N.º X", que usamos como fronteira de chunk.

Uso:  python src/ingest.py
"""
import re
import sys
import unicodedata
from pathlib import Path

from common import load_config, write_jsonl

# ---------------------------------------------------------------- datas por extenso
NUM = {
    "um": 1, "uma": 1, "primeiro": 1, "dois": 2, "duas": 2, "tres": 3, "quatro": 4,
    "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9, "dez": 10, "onze": 11,
    "doze": 12, "treze": 13, "catorze": 14, "quatorze": 14, "quinze": 15,
    "dezasseis": 16, "dezesseis": 16, "dezassete": 17, "dezessete": 17, "dezoito": 18,
    "dezanove": 19, "dezenove": 19, "vinte": 20, "trinta": 30, "quarenta": 40,
    "cinquenta": 50, "sessenta": 60, "setenta": 70, "oitenta": 80, "noventa": 90,
}
MESES = {
    "janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
    "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}


def sem_acentos(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def palavras_para_numero(tokens: list[str]) -> int | None:
    """['dois','mil','e','dezassete'] -> 2017 ; ['trinta','e','um'] -> 31"""
    total, atual, viu = 0, 0, False
    for w in tokens:
        if w == "e":
            continue
        if w.isdigit():
            atual += int(w); viu = True
        elif w == "mil":
            total += (atual or 1) * 1000; atual = 0; viu = True
        elif w in NUM:
            atual += NUM[w]; viu = True
        else:
            return None
    return total + atual if viu else None


DATA_RE = re.compile(
    r"((?:\b[a-z0-9]+\b[\s-]+){1,4}?)(?:dias?\s+)?(?:do\s+mes\s+)?de\s+("
    + "|".join(MESES) + r")\s+(?:do\s+ano\s+)?de\s+((?:\b[a-z0-9]+\b\s*){1,7})"
)


def extrair_data(texto: str) -> str | None:
    """Devolve a 1.ª data encontrada no texto em formato ISO (AAAA-MM-DD)."""
    t = re.sub(r"\s+", " ", sem_acentos(texto.lower()))
    for m in DATA_RE.finditer(t):
        # dia: últimos tokens numéricos antes do mês
        toks = m.group(1).split()
        dia_toks = []
        for w in reversed(toks):
            if w in NUM or w.isdigit() or w == "e":
                dia_toks.insert(0, w)
            else:
                break
        # ano: tokens numéricos depois do mês
        ano_toks = []
        for w in m.group(3).split():
            if w in NUM or w.isdigit() or w in ("mil", "e"):
                ano_toks.append(w)
            else:
                break
        while ano_toks and ano_toks[-1] == "e":
            ano_toks.pop()
        dia, ano = palavras_para_numero(dia_toks), palavras_para_numero(ano_toks)
        if dia and ano and 1 <= dia <= 31 and 1970 <= ano <= 2100:
            return f"{ano:04d}-{MESES[m.group(2)]:02d}-{dia:02d}"
    return None


# ---------------------------------------------------------------- limpeza
def limpar(texto: str) -> str:
    texto = re.sub(r"-{3,}", " ", texto)                         # tracejados
    texto = re.sub(r"(?m)^\s*\d{1,3}\s*$", "", texto)             # n.º de página
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)                # hifenização
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n\s*\n+", "\n\n", texto)
    return texto.strip()


def metadados(texto: str, ficheiro: str) -> dict:
    inicio = texto[:3000]
    n = re.search(r"(?i)(?:^|\s)a?c?ta\s+n\.?\s*[º°o]\.?\s*(\d+)", inicio)
    tipo = "extraordinária" if re.search(r"(?i)extraordin", inicio) else "ordinária"
    return {
        "ficheiro": ficheiro,
        "ata_numero": int(n.group(1)) if n else None,
        "data_reuniao": extrair_data(inicio),
        "tipo_reuniao": tipo,
    }


# ---------------------------------------------------------------- chunking
PROPOSTA_RE = re.compile(r"(?i)(?=\bproposta\s+n\.?\s*[º°o]\.?\s*\d+)")


def janela(texto: str, max_chars: int, overlap: int) -> list[str]:
    if len(texto) <= max_chars:
        return [texto]
    partes, i = [], 0
    while i < len(texto):
        fim = min(i + max_chars, len(texto))
        if fim < len(texto):  # cortar no fim de frase mais próximo
            corte = texto.rfind(". ", i + max_chars // 2, fim)
            fim = corte + 1 if corte != -1 else fim
        partes.append(texto[i:fim].strip())
        if fim >= len(texto):
            break
        i = max(fim - overlap, i + 1)
    return partes


def chunkar(texto: str, meta: dict, max_chars: int, overlap: int) -> list[dict]:
    cabecalho = (f"[Ata n.º {meta['ata_numero'] or '?'} | reunião {meta['tipo_reuniao']} "
                 f"| {meta['data_reuniao'] or 'data desconhecida'}]")
    chunks = []
    for secao in PROPOSTA_RE.split(texto):
        secao = secao.strip()
        if not secao:
            continue
        p = re.match(r"(?i)proposta\s+n\.?\s*[º°o]\.?\s*(\d+)", secao)
        for parte in janela(secao, max_chars, overlap):
            chunks.append({
                "id": f"{Path(meta['ficheiro']).stem}_{len(chunks):04d}",
                # o cabeçalho entra no texto indexado: ajuda perguntas com datas/n.º de ata
                "text": f"{cabecalho}\n{parte}",
                "metadata": {**meta, "proposta": int(p.group(1)) if p else -1},
            })
    return chunks


def ler_pdf(path: Path) -> str:
    import fitz  # pymupdf

    with fitz.open(path) as doc:
        return "\n".join(page.get_text() for page in doc)


def main():
    cfg = load_config()
    pdfs = sorted(Path(cfg["paths"]["atas_dir"]).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"Nenhum PDF em {cfg['paths']['atas_dir']}")
    todos = []
    for pdf in pdfs:
        texto = limpar(ler_pdf(pdf))
        if len(texto) < 200:
            print(f"AVISO: {pdf.name} tem pouco texto — PDF digitalizado? Precisa de OCR.")
            continue
        meta = metadados(texto, pdf.name)
        ch = chunkar(texto, meta, cfg["chunking"]["max_chars"], cfg["chunking"]["overlap_chars"])
        print(f"{pdf.name}: ata {meta['ata_numero']}, {meta['data_reuniao']}, {len(ch)} chunks")
        todos.extend(ch)
    write_jsonl(cfg["paths"]["chunks"], todos)
    print(f"\n{len(todos)} chunks -> {cfg['paths']['chunks']}")


if __name__ == "__main__":
    main()
