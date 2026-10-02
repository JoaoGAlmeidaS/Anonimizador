"""
Mede a censura sobre os textos fictícios de `tests/fixtures/` e grava
`docs/medicoes.md`.

    python ferramentas/medir_censura.py pt_core_news_md pt_core_news_lg

Cada argumento é um modelo spaCy: nome do pacote instalado ou caminho da pasta.
Para cada um: recall por camada, o que escapou, o que foi censurado sem
necessidade, tempo de carga e tempo de ~60 minutos de fala (~8 mil palavras).
"""
import os
import platform
import sys
import time
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from anonimizador import anonimizar  # noqa: E402
from anonimizador.motor import MotorCensura, carregar_spacy  # noqa: E402
from tests.avaliacao import avaliar, carregar_fixtures, recall_por_camada  # noqa: E402

DESTINO = RAIZ / "docs" / "medicoes.md"
NOMES_DAS_CAMADAS = {1: "1 (deny-list)", 2: "2 (regex)", 3: "3 (IBGE)", 4: "4 (só NER)"}


def medir(modelo: str) -> dict:
    inicio = time.perf_counter()
    nlp = carregar_spacy(modelo)
    motor = MotorCensura(nlp)
    carga = time.perf_counter() - inicio

    fixtures = carregar_fixtures()
    linhas, escaparam, censurados_sem_necessidade = [], [], []
    for fixture in fixtures:
        resultado = anonimizar(fixture["texto"], fixture["paciente"], motor=motor)
        avaliadas = avaliar(fixture, resultado)
        linhas += avaliadas
        escaparam += [f"{l['arquivo']}: “{l['trecho']}” (camada {l['camada']})" for l in avaliadas if not l["coberto"]]
        censurados_sem_necessidade += [
            f"{fixture['arquivo']}: “{trecho}”"
            for trecho in fixture.get("preservar", []) + fixture.get("preservar_com_ner", [])
            if trecho not in resultado.texto_censurado
        ]

    # ~130 palavras por minuto de fala → ~8 mil palavras em 60 minutos.
    longo = next(f for f in fixtures if f["arquivo"].startswith("22_"))
    texto = "\n".join([longo["texto"]] * (8000 // len(longo["texto"].split()) + 1))
    inicio = time.perf_counter()
    anonimizar(texto, longo["paciente"], motor=motor)
    tempo_60min = time.perf_counter() - inicio

    return {
        # Nome do pacote, mesmo quando o modelo veio de uma pasta (o caminho é local).
        "modelo": f"{nlp.meta['lang']}_{nlp.meta['name']} {nlp.meta['version']}",
        "recall": recall_por_camada(linhas),
        "escaparam": escaparam,
        "censurados_sem_necessidade": censurados_sem_necessidade,
        "carga_s": carga,
        "palavras_60min": len(texto.split()),
        "tempo_60min_s": tempo_60min,
    }


def relatorio(medidas: list[dict]) -> str:
    import spacy

    camadas = sorted({camada for medida in medidas for camada in medida["recall"]})
    linhas = [
        "# Medições",
        "",
        "Geradas por `ferramentas/medir_censura.py` — não editar à mão; rodar de novo.",
        "",
        f"- Data: {date.today().isoformat()}",
        f"- Máquina: {platform.platform()}, {os.cpu_count()} threads; Python {platform.python_version()}, spaCy {spacy.__version__}",
        f"- Textos: os {len(carregar_fixtures())} fixtures fictícios de `tests/fixtures/`",
        "",
        "Recall = ocorrências censuradas / ocorrências esperadas, pela camada que tem a",
        "obrigação de pegar o caso. \"60 min de fala\" = o fixture 22 repetido até ~8 mil palavras.",
        "",
        "| Modelo | " + " | ".join(f"Camada {NOMES_DAS_CAMADAS.get(c, c)}" for c in camadas)
        + " | 60 min de fala | Carga |",
        "|---|" + "---|" * len(camadas) + "---|---|",
    ]
    for medida in medidas:
        recall = " | ".join(
            "{}/{}".format(*medida["recall"].get(camada, (0, 0))) for camada in camadas
        )
        linhas.append(
            f"| {medida['modelo']} | {recall} | {medida['tempo_60min_s']:.2f} s "
            f"({medida['palavras_60min']} palavras) | {medida['carga_s']:.1f} s |"
        )
    linhas.append("")
    for medida in medidas:
        linhas.append(f"**{medida['modelo']} — escaparam:** " + ("; ".join(medida["escaparam"]) or "nada"))
        linhas.append("")
        linhas.append(
            f"**{medida['modelo']} — censurados sem necessidade:** "
            + ("; ".join(medida["censurados_sem_necessidade"]) or "nada")
        )
        linhas.append("")
    return "\n".join(linhas)


def main() -> None:
    modelos = sys.argv[1:]
    if not modelos:
        sys.exit(__doc__)
    medidas = []
    for modelo in modelos:
        print(f"medindo {modelo}…", flush=True)
        medidas.append(medir(modelo))
    DESTINO.parent.mkdir(exist_ok=True)
    DESTINO.write_text(relatorio(medidas), encoding="utf-8")
    print(f"gravado em {DESTINO.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
