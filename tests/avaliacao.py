"""
Textos fictícios anotados do anonimizador (`tests/fixtures/*.json`) e a conta
de recall por camada. Usado pelos testes e por `ferramentas/medir_censura.py`.

Formato de cada fixture:
    paciente   — dados fictícios que alimentam a deny-list
    texto      — o texto a censurar
    esperado   — [{trecho, tipo, camada}]: toda ocorrência de `trecho` precisa sair
                 censurada. `camada` é a que tem a obrigação de pegar o caso:
                 1 deny-list, 2 regex, 3 nomes IBGE, 4 só o NER pega.
    preservar  — trechos que precisam continuar no texto censurado, com ou sem o NER
    preservar_com_ner — idem, mas só conferidos com o modelo de verdade: dependem
                 do léxico de frequência dele (ex.: "tomei leite" quando o
                 paciente se chama Leite)

Nenhum dado aqui é de pessoa real: nomes, documentos e contatos são inventados.
"""
import json
import re
from pathlib import Path

PASTA = Path(__file__).parent / "fixtures"


def carregar_fixtures() -> list[dict]:
    fixtures = []
    for arquivo in sorted(PASTA.glob("*.json")):
        fixture = json.loads(arquivo.read_text(encoding="utf-8"))
        fixture["arquivo"] = arquivo.name
        fixtures.append(fixture)
    return fixtures


def ocorrencias(texto: str, trecho: str) -> list[tuple[int, int]]:
    padrao = r"(?<!\w)" + re.escape(trecho) + r"(?!\w)"
    return [(achado.start(), achado.end()) for achado in re.finditer(padrao, texto)]


def esta_coberto(texto: str, inicio: int, fim: int, trechos: list[dict]) -> bool:
    """
    Toda letra e todo dígito da ocorrência caem dentro de algum trecho censurado.
    Separadores podem ficar de fora: "Ana-Luísa" → "[PACIENTE]-[PACIENTE]" não vaza nada.
    """
    return all(
        any(censurado["inicio"] <= posicao < censurado["fim"] for censurado in trechos)
        for posicao in range(inicio, fim)
        if texto[posicao].isalnum()
    )


def avaliar(fixture: dict, resultado) -> list[dict]:
    """Uma linha por ocorrência esperada: camada, trecho e se saiu censurada."""
    linhas = []
    for esperado in fixture["esperado"]:
        achados = ocorrencias(fixture["texto"], esperado["trecho"])
        assert achados, f"{fixture['arquivo']}: '{esperado['trecho']}' não está no texto"
        for inicio, fim in achados:
            linhas.append(
                {
                    "arquivo": fixture["arquivo"],
                    "camada": esperado["camada"],
                    "trecho": esperado["trecho"],
                    "coberto": esta_coberto(fixture["texto"], inicio, fim, resultado.trechos),
                }
            )
    return linhas


def recall_por_camada(linhas: list[dict]) -> dict[int, tuple[int, int]]:
    """camada → (cobertas, total)."""
    contas: dict[int, list[int]] = {}
    for linha in linhas:
        conta = contas.setdefault(linha["camada"], [0, 0])
        conta[0] += int(linha["coberto"])
        conta[1] += 1
    return {camada: tuple(conta) for camada, conta in sorted(contas.items())}
