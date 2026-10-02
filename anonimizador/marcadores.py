"""
Troca dos trechos sensíveis por marcadores — `[PACIENTE]`, `[PESSOA_1]`,
`[LOCAL_1]`... — e o caminho de volta (`reverter`).

Feito aqui, e não no presidio-anonymizer, por três motivos: ele prende o
`cryptography` abaixo da versão 49 (o que conflita com quem usa uma versão nova
para outra coisa), o operador customizado dele recebe só o texto (sem tipo nem
posição) e a resolução de conflitos dele reembaralharia os trechos que já
resolvemos aqui.

Este módulo é Python puro: `reverter` funciona sem carregar spaCy nem Presidio.
"""
import re
from collections import Counter
from dataclasses import dataclass

from anonimizador.texto import chave

# Tipos de entidade, já com o nome do marcador. O próprio paciente não é numerado.
PACIENTE = "PACIENTE"
TIPOS = ("PACIENTE", "PESSOA", "LOCAL", "ORGANIZACAO", "DATA", "CPF", "TELEFONE", "EMAIL", "CEP")
# Tipos com formato próprio, vindos das camadas de regex e da deny-list.
ESTRUTURADOS = {"DATA", "CPF", "TELEFONE", "EMAIL", "CEP"}


@dataclass(frozen=True)
class Trecho:
    """Um trecho detectado. `camada`: 1 deny-list, 2 regex, 3 nomes IBGE, 4 NER."""

    inicio: int
    fim: int
    tipo: str
    camada: int
    score: float

    @property
    def tamanho(self) -> int:
        return self.fim - self.inicio


# --- Sobreposições ---

def resolver_sobreposicoes(trechos: list[Trecho]) -> list[Trecho]:
    """
    Funde os trechos que se sobrepõem num trecho só, do início do primeiro ao fim
    do último — censurar a união nunca deixa escapar um pedaço.

    O tipo do trecho fundido:
    - um dado estruturado (e-mail, CPF, telefone, CEP, data) que ocupa o trecho
      inteiro vence: o e-mail do paciente contém o nome dele, mas é um e-mail;
    - senão, é `[PACIENTE]` quando os pedaços do paciente (deny-list) cobrem mais
      da metade do trecho. Sem isso, um NER que pegasse "Dona Ana Souza" (maior que
      o "Ana Souza" da deny-list) transformaria o próprio paciente num
      `[PESSOA_3]`; e sem a metade, um terceiro com o mesmo sobrenome
      ("Carlos Souza") viraria `[PACIENTE]`;
    - senão, vale o pedaço mais longo, desempatando pela
      camada mais confiável e depois pelo score. A camada vem antes do score
      porque o rótulo do NER é ruidoso: o `md` chama "Juliana" de lugar, enquanto
      o dicionário de nomes (camada 3) sabe que é pessoa.
    """
    grupos: list[list[Trecho]] = []
    fim_do_grupo = -1
    for trecho in sorted(trechos, key=lambda t: (t.inicio, -t.fim)):
        if grupos and trecho.inicio < fim_do_grupo:
            grupos[-1].append(trecho)
            fim_do_grupo = max(fim_do_grupo, trecho.fim)
        else:
            grupos.append([trecho])
            fim_do_grupo = trecho.fim

    return [_fundir(grupo) for grupo in grupos]


def _cobertura(trechos: list[Trecho]) -> int:
    """Quantos caracteres os trechos cobrem juntos, sem contar duas vezes."""
    coberto, fim_atual = 0, -1
    for trecho in sorted(trechos, key=lambda t: t.inicio):
        inicio = max(trecho.inicio, fim_atual)
        if trecho.fim > inicio:
            coberto += trecho.fim - inicio
        fim_atual = max(fim_atual, trecho.fim)
    return coberto


def _fundir(grupo: list[Trecho]) -> Trecho:
    inicio = min(trecho.inicio for trecho in grupo)
    fim = max(trecho.fim for trecho in grupo)

    pacientes = [trecho for trecho in grupo if trecho.tipo == PACIENTE]
    outros = [trecho for trecho in grupo if trecho.tipo != PACIENTE]
    estruturados_inteiros = [
        trecho for trecho in outros
        if trecho.tipo in ESTRUTURADOS and trecho.inicio == inicio and trecho.fim == fim
    ]
    if estruturados_inteiros:
        tipo = max(estruturados_inteiros, key=lambda t: (t.score, -t.camada)).tipo
    elif pacientes and (not outros or 2 * _cobertura(pacientes) > fim - inicio):
        tipo = PACIENTE
    else:
        tipo = max(outros, key=lambda t: (t.tamanho, -t.camada, t.score)).tipo

    return Trecho(
        inicio=inicio,
        fim=fim,
        tipo=tipo,
        camada=min(trecho.camada for trecho in grupo),
        score=max(trecho.score for trecho in grupo),
    )


# --- Aplicação dos marcadores ---

@dataclass
class Aplicacao:
    texto_censurado: str
    mapa: dict[str, str]
    trechos: list[dict]
    segmentos: list[dict]
    contagem: dict[str, int]


def aplicar(texto: str, trechos: list[Trecho]) -> Aplicacao:
    """
    Monta o texto censurado a partir de trechos já sem sobreposição.

    O mesmo valor recebe sempre o mesmo marcador dentro do texto, e a numeração
    segue a ordem de aparição. O mapa guarda a primeira grafia de cada marcador.

    `segmentos` é o texto original fatiado em pedaços comuns e censurados, para
    uma tela de revisão destacar os trechos sem fazer conta de posição em JS
    (índice do Python ≠ índice UTF-16 do JS quando há emoji).
    """
    marcador_por_valor: dict[tuple[str, str], str] = {}
    numeracao: Counter = Counter()
    mapa: dict[str, str] = {}
    saida, detalhes, segmentos = [], [], []
    cursor = 0

    for trecho in sorted(trechos, key=lambda t: t.inicio):
        valor = texto[trecho.inicio:trecho.fim]
        if trecho.tipo == PACIENTE:
            marcador = f"[{PACIENTE}]"
        else:
            identidade = (trecho.tipo, chave(valor))
            if identidade not in marcador_por_valor:
                numeracao[trecho.tipo] += 1
                marcador_por_valor[identidade] = f"[{trecho.tipo}_{numeracao[trecho.tipo]}]"
            marcador = marcador_por_valor[identidade]
        mapa.setdefault(marcador, valor)

        if cursor < trecho.inicio:
            comum = texto[cursor:trecho.inicio]
            saida.append(comum)
            segmentos.append({"texto": comum, "tipo": None, "marcador": None})
        saida.append(marcador)
        segmentos.append({"texto": valor, "tipo": trecho.tipo, "marcador": marcador})
        detalhes.append(
            {
                "inicio": trecho.inicio,
                "fim": trecho.fim,
                "tipo": trecho.tipo,
                "camada": trecho.camada,
                "marcador": marcador,
            }
        )
        cursor = trecho.fim

    if cursor < len(texto):
        saida.append(texto[cursor:])
        segmentos.append({"texto": texto[cursor:], "tipo": None, "marcador": None})

    # Contagem = marcadores distintos por tipo ("3 pessoas", não "3 menções").
    contagem = Counter(marcador.strip("[]").split("_")[0] for marcador in mapa)
    return Aplicacao(
        texto_censurado="".join(saida),
        mapa=mapa,
        trechos=detalhes,
        segmentos=segmentos,
        contagem=dict(contagem),
    )


# --- Reversão ---

_NOMES_TIPO = "|".join(TIPOS) + "|ORGANIZAÇÃO|E-MAIL"

# Com colchetes, aceita qualquer caixa e separador: [PESSOA_1], [Pessoa 1], [pessoa-1].
_COM_COLCHETES = re.compile(
    rf"\[\s*({_NOMES_TIPO})(?:[\s_-]*(\d+))?\s*\]", re.IGNORECASE
)
# Sem colchetes, só com número (PESSOA_1, Pessoa_1). Sem número e sem colchetes,
# "pessoa", "local", "data" e "paciente" são palavras comuns do texto.
_SEM_COLCHETES = re.compile(rf"(?<![\w\[])({_NOMES_TIPO})_(\d+)(?![\w\]])", re.IGNORECASE)
# PACIENTE nu só em caixa alta — "o paciente relatou" fica como está.
_PACIENTE_NU = re.compile(r"(?<![\w\[])PACIENTE(?![\w\]])")


def _canonico(tipo: str, numero: str | None) -> str:
    tipo = tipo.upper().replace("Ç", "C").replace("Ã", "A").replace("-", "")
    return f"[{tipo}]" if numero is None else f"[{tipo}_{int(numero)}]"


def reverter(texto: str, mapa: dict[str, str]) -> str:
    """
    Devolve os valores originais no lugar dos marcadores. Tolerante ao que o
    modelo de fronteira costuma devolver (`PESSOA_1` sem colchetes, `[Pessoa_1]`).
    Marcador que não está no mapa fica como veio.
    """

    def trocar(encontrado: re.Match) -> str:
        grupos = encontrado.groups()
        canonico = _canonico(grupos[0], grupos[1] if len(grupos) > 1 else None)
        return mapa.get(canonico, encontrado.group(0))

    texto = _COM_COLCHETES.sub(trocar, texto)
    texto = _SEM_COLCHETES.sub(trocar, texto)
    if f"[{PACIENTE}]" in mapa:
        texto = _PACIENTE_NU.sub(lambda _: mapa[f"[{PACIENTE}]"], texto)
    return texto
