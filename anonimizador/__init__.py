"""
Censura local: troca dados que identificam pessoas por marcadores, antes de
qualquer texto sair da máquina (por exemplo, para uma LLM na nuvem).

API pública:

    anonimizar(texto, dados_paciente) -> ResultadoAnonimizacao
    reverter(texto, mapa) -> str

O mapa (marcador → valor original) só existe em memória: nunca é persistido,
logado nem enviado. Os campos com conteúdo clínico do resultado têm `repr=False`
para um `print(resultado)` distraído não despejar o texto clínico num log.

O Presidio e o spaCy são importados só dentro de `anonimizar`: `reverter` e os
marcadores funcionam sem eles.
"""
import logging
import time
from dataclasses import dataclass, field

from anonimizador.marcadores import aplicar, resolver_sobreposicoes, reverter

__all__ = ["ResultadoAnonimizacao", "anonimizar", "reverter"]

logger = logging.getLogger(__name__)


@dataclass
class ResultadoAnonimizacao:
    texto_censurado: str = field(repr=False)
    # marcador → valor original. Só em memória: nunca persistir, logar nem enviar.
    mapa: dict[str, str] = field(repr=False)
    # posições no texto original, tipo, camada e marcador — para a revisão humana.
    trechos: list[dict] = field(repr=False)
    # o original fatiado em pedaços comuns e censurados, pronto para destacar.
    segmentos: list[dict] = field(repr=False)
    # marcadores distintos por tipo. Metadado: pode ir para o log.
    contagem: dict[str, int] = field(default_factory=dict)


def anonimizar(texto: str, dados_paciente: dict, *, motor=None) -> ResultadoAnonimizacao:
    """
    `dados_paciente` alimenta a deny-list (camada 1): nome, cpf, telefone, email e
    contato_emergencia. `motor` existe para os testes injetarem um pipeline sem
    modelo; por padrão, é o motor com o modelo spaCy instalado (ver `motor.obter_motor`).
    """
    if motor is None:
        from anonimizador.motor import obter_motor

        motor = obter_motor()

    inicio = time.perf_counter()
    trechos = resolver_sobreposicoes(motor.analisar(texto, dados_paciente))
    aplicacao = aplicar(texto, trechos)
    logger.info(
        "censura: %d caracteres, %d trechos, contagem=%s, %.2fs",
        len(texto),
        len(trechos),
        aplicacao.contagem,
        time.perf_counter() - inicio,
    )
    return ResultadoAnonimizacao(
        texto_censurado=aplicacao.texto_censurado,
        mapa=aplicacao.mapa,
        trechos=aplicacao.trechos,
        segmentos=aplicacao.segmentos,
        contagem=aplicacao.contagem,
    )
