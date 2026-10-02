"""
Gera a lista de nomes usada pela camada 3 do anonimizador.

    pip install -e ".[ferramentas]"
    python ferramentas/gerar_nomes_ibge.py

Fonte: API de Nomes do IBGE (Censo Demográfico 2010),
https://servicodados.ibge.gov.br/api/docs/nomes?versao=2

A API não exporta a base inteira. O script junta duas fontes, ambas do IBGE:

1. o ranking (top 20 por consulta), cruzando os filtros que ela oferece — Brasil e
   cada UF × sexo × década de nascimento;
2. a consulta de frequência por nome, aplicada aos candidatos escritos à mão em
   `nomes_candidatos.txt`. Candidato que o Censo não confirma (ou que tem menos de
   FREQUENCIA_MINIMA pessoas) fica de fora.

O resultado não é a base inteira do Censo. Os nomes raros ficam por conta do NER
do spaCy (camada 4) e da revisão humana.

O arquivo gerado é versionado; este script só roda de novo para atualizá-lo.
"""
import time
from datetime import date
from pathlib import Path

import httpx

URL = "https://servicodados.ibge.gov.br/api/v2/censos/nomes/ranking"
URL_POR_NOME = "https://servicodados.ibge.gov.br/api/v2/censos/nomes/"
CANDIDATOS = Path(__file__).resolve().parent / "nomes_candidatos.txt"
FREQUENCIA_MINIMA = 200
DESTINO = Path(__file__).resolve().parent.parent / "anonimizador" / "dados" / "nomes_ibge.txt"

# Brasil inteiro (None) e as 27 UFs, pelos códigos do IBGE.
LOCALIDADES = [None, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 28, 29,
               31, 32, 33, 35, 41, 42, 43, 50, 51, 52, 53]
SEXOS = [None, "M", "F"]
DECADAS = [None, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000]


def consultar(cliente: httpx.Client, localidade, sexo, decada) -> list[tuple[str, int]]:
    parametros = {
        chave: valor
        for chave, valor in (("localidade", localidade), ("sexo", sexo), ("decada", decada))
        if valor is not None
    }
    for tentativa in range(3):
        try:
            resposta = cliente.get(URL, params=parametros)
            resposta.raise_for_status()
            corpo = resposta.json()
            if not corpo:
                return []
            return [(item["nome"], item["frequencia"]) for item in corpo[0]["res"]]
        except (httpx.HTTPError, ValueError, KeyError):
            time.sleep(2 * (tentativa + 1))
    print(f"  falhou: {parametros}")
    return []


def frequencias(cliente: httpx.Client, candidatos: list[str]) -> dict[str, int]:
    """Frequência total no Censo de cada candidato. Os que o IBGE não conhece somem."""
    resultado: dict[str, int] = {}
    for inicio in range(0, len(candidatos), 20):
        lote = candidatos[inicio:inicio + 20]
        for tentativa in range(3):
            try:
                resposta = cliente.get(URL_POR_NOME + "|".join(lote))
                resposta.raise_for_status()
                for item in resposta.json():
                    resultado[item["nome"]] = sum(periodo["frequencia"] for periodo in item["res"])
                break
            except (httpx.HTTPError, ValueError, KeyError):
                time.sleep(2 * (tentativa + 1))
        time.sleep(0.15)
    return resultado


def ler_candidatos() -> list[str]:
    palavras = []
    for linha in CANDIDATOS.read_text(encoding="utf-8").splitlines():
        if linha.strip() and not linha.startswith("#"):
            palavras.extend(linha.split())
    return sorted(set(palavras))


def main() -> None:
    nomes: dict[str, int] = {}
    total = len(LOCALIDADES) * len(SEXOS) * len(DECADAS)
    feitas = 0
    with httpx.Client(timeout=30) as cliente:
        for localidade in LOCALIDADES:
            for sexo in SEXOS:
                for decada in DECADAS:
                    for nome, frequencia in consultar(cliente, localidade, sexo, decada):
                        nomes[nome] = max(nomes.get(nome, 0), frequencia)
                    feitas += 1
                    if feitas % 50 == 0:
                        print(f"{feitas}/{total} consultas, {len(nomes)} nomes")
                    # Gentileza com o servidor público: nada de rajada.
                    time.sleep(0.15)
        do_ranking = len(nomes)

        candidatos = ler_candidatos()
        confirmados = {
            nome: frequencia
            for nome, frequencia in frequencias(cliente, candidatos).items()
            if frequencia >= FREQUENCIA_MINIMA
        }
        print(f"{len(confirmados)} de {len(candidatos)} candidatos confirmados pelo IBGE")
        for nome, frequencia in confirmados.items():
            nomes[nome] = max(nomes.get(nome, 0), frequencia)

    cabecalho = [
        "# Nomes próprios para a camada 3 do anonimizador (um por linha, maiúsculo, sem acento).",
        "# Fonte: IBGE, Censo Demográfico 2010 — API de Nomes v2",
        "#   https://servicodados.ibge.gov.br/api/docs/nomes?versao=2",
        f"# {do_ranking} nomes da união dos rankings (top 20) por Brasil/UF × sexo × década;",
        f"# o resto são candidatos de ferramentas/nomes_candidatos.txt confirmados pela consulta",
        f"# de frequência do IBGE (mínimo de {FREQUENCIA_MINIMA} pessoas no Censo).",
        f"# Gerada em {date.today().isoformat()} por ferramentas/gerar_nomes_ibge.py — {len(nomes)} nomes.",
        "# Ao reutilizar esta lista, cite a fonte (IBGE).",
    ]
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    DESTINO.write_text("\n".join(cabecalho + sorted(nomes)) + "\n", encoding="utf-8")
    print(f"{len(nomes)} nomes gravados em {DESTINO}")


if __name__ == "__main__":
    main()
