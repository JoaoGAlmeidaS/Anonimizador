"""
Reconhecedores do Presidio para português, em camadas:

1. deny-list do paciente — montada a cada chamada, a partir dos dados dele;
2. padrões brasileiros por regex — CPF, telefone, CEP, e-mail, datas;
3. dicionário de nomes do IBGE;
4. NER do spaCy, filtrado contra palavras comuns com maiúscula (o pipeline é
   carregado em `motor.py`).

Os reconhecedores pré-definidos do Presidio NÃO são usados: são voltados a inglês
e a formatos dos EUA — e alguns (e-mail, URL) usam o tldextract, que baixa a lista
de sufixos pela internet.

Todo resultado sai com o tipo já no nome do marcador (`PESSOA`, `LOCAL`...).
"""
import re
from functools import lru_cache
from pathlib import Path

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.predefined_recognizers import SpacyRecognizer

from anonimizador.lexico import Lexico
from anonimizador.texto import chave, normalizar

IDIOMA = "pt"

# O Presidio liga DOTALL | MULTILINE | IGNORECASE por padrão. Aqui as flags são
# explícitas: nenhum padrão precisa de DOTALL, e IGNORECASE só importa para as
# palavras ("arroba", "CEP", nomes de mês).
FLAGS = re.IGNORECASE

PASTA_DADOS = Path(__file__).parent / "dados"

# Nome do reconhecedor → camada, para o relatório de recall e para desempate.
CAMADA_POR_RECONHECEDOR = {
    "DenyListPaciente": 1,
    "Cpf": 2,
    "Telefone": 2,
    "Cep": 2,
    "Email": 2,
    "EmailFalado": 2,
    "Data": 2,
    "NomesIbge": 3,
    "NerSpacy": 4,
}

# Partes de nome que não identificam ninguém.
PREPOSICOES = {"de", "da", "do", "das", "dos", "e", "d", "di", "du", "del", "della",
               "van", "von", "der", "la", "le", "y"}

# Palavras que aparecem no campo livre `contato_emergencia` sem ser nome.
PALAVRAS_DE_CONTATO = {
    "mae", "pai", "irma", "irmao", "esposa", "esposo", "marido", "mulher", "filho",
    "filha", "tia", "tio", "avo", "avos", "vovo", "prima", "primo", "sogra", "sogro",
    "cunhada", "cunhado", "namorada", "namorado", "noiva", "noivo", "companheira",
    "companheiro", "amiga", "amigo", "vizinha", "vizinho", "madrasta", "padrasto",
    "enteada", "enteado", "neta", "neto", "responsavel", "contato", "emergencia",
    "tel", "telefone", "celular", "cel", "fone", "whatsapp", "zap", "email", "mail",
    "casa", "trabalho", "recado",
}


def _resultado(reconhecedor: EntityRecognizer, tipo: str, inicio: int, fim: int, score: float):
    return RecognizerResult(
        entity_type=tipo,
        start=inicio,
        end=fim,
        score=score,
        recognition_metadata={
            RecognizerResult.RECOGNIZER_NAME_KEY: reconhecedor.name,
            RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: reconhecedor.id,
        },
    )


def _digitos_com_separadores(digitos: str) -> str:
    """'11987' → regex que aceita '11 987', '(11) 9-87'... entre os dígitos."""
    return r"(?<!\d)" + r"[\s().-]*".join(digitos) + r"(?!\d)"


def _palavra(normalizado: str) -> str:
    return r"(?<!\w)" + re.escape(normalizado) + r"(?!\w)"


# --- Camada 1: deny-list do paciente ---

def _partes_de_nome(texto: str) -> list[str]:
    """Palavras com letras (acentos, hífen e apóstrofo inclusos)."""
    return re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", texto or "")


def _variantes_de_telefone(digitos: str) -> set[str]:
    """
    O número como foi cadastrado e o que costuma ser dito: sem o +55 e sem o DDD.
    Menos de 8 dígitos não é telefone, e casar números curtos censuraria o texto todo.
    """
    variantes = set()
    if len(digitos) in (12, 13) and digitos.startswith("55"):
        digitos = digitos[2:]
    if len(digitos) >= 8:
        variantes.add(digitos)
    if len(digitos) in (10, 11):
        variantes.add(digitos[2:])
    return variantes


def _email_falado(email: str) -> str:
    """'ana.souza@gmail.com' → regex de 'ana ponto souza arroba gmail ponto com'."""
    falado = {"@": "arroba", ".": "ponto", "_": "underline", "-": "(?:tracinho|traco|hifen)"}
    partes = re.split(r"([@._-])", normalizar(email))
    pedacos = [falado[parte] if parte in falado else re.escape(parte) for parte in partes if parte]
    return r"(?<!\w)" + r"\s+".join(pedacos) + r"(?!\w)"


# Parte de nome que também é palavra comum ("Leite", "Filho") só conta sozinha se
# vier com maiúscula. Mesmo limiar das palavras comuns do NER (ver lexico.py).
LIMIAR_PALAVRA_COMUM = 1.5
_CONECTIVO_OPCIONAL = r"(?:\s+(?:de|da|do|das|dos|e))?\s+"


def _termos_de_nome(partes: list[str], tipo: str, lexico) -> list[tuple[str, str, bool]]:
    """
    O nome inteiro, cada par de partes na ordem em que aparecem (as pessoas pulam
    nomes do meio: "joão leite" para "João Pedro Leite") e cada parte sozinha.

    Nome inteiro e pares valem sem diferenciar maiúsculas: uma transcrição
    automática às vezes escreve tudo minúsculo. Já uma parte sozinha que também é
    palavra comum ("Leite", "Campos") ou nome ambíguo ("Filho", "Neto") só conta
    com maiúscula: sem isso, "tomei leite" virava "tomei [PACIENTE]".
    """
    termos: list[tuple[str, str, bool]] = []
    if len(partes) >= 2:
        termos.append((r"(?<!\w)" + r"\s+".join(map(re.escape, partes)) + r"(?!\w)", tipo, False))

    significativas = [parte for parte in partes if len(parte) >= 3 and parte not in PREPOSICOES]
    for i, primeira in enumerate(significativas):
        for segunda in significativas[i + 1:]:
            par = re.escape(primeira) + _CONECTIVO_OPCIONAL + re.escape(segunda)
            termos.append((r"(?<!\w)" + par + r"(?!\w)", tipo, False))

    ambiguos = nomes_ambiguos()
    for parte in significativas:
        sensivel = parte in ambiguos or lexico.eh_palavra_comum(parte, LIMIAR_PALAVRA_COMUM)
        termos.append((_palavra(parte), tipo, sensivel))
    return termos


def montar_termos(dados_paciente: dict, lexico=None) -> list[tuple[re.Pattern, str, bool]]:
    """
    Converte os dados do paciente em padrões já normalizados (sem acento,
    minúsculos), cada um com o tipo de marcador que vai receber e se exige
    inicial maiúscula no texto original.
    """
    lexico = lexico or Lexico.vazio()
    termos: list[tuple[str, str, bool]] = []

    # Partes do nome do paciente viram [PACIENTE].
    partes = [normalizar(parte) for parte in _partes_de_nome(dados_paciente.get("nome"))]
    termos += _termos_de_nome(partes, "PACIENTE", lexico)

    # Contato de emergência: texto livre ("Joana Lima (mãe) 11 91234-5678").
    contato = dados_paciente.get("contato_emergencia") or ""
    nomes_contato = [
        normalizar(parte) for parte in _partes_de_nome(contato)
        if len(parte) >= 3
        and normalizar(parte) not in PREPOSICOES
        and normalizar(parte) not in PALAVRAS_DE_CONTATO
    ]
    termos += _termos_de_nome(nomes_contato, "PESSOA", lexico)
    for numero in re.findall(r"\+?\d[\d\s().-]{6,}\d", contato):
        for variante in _variantes_de_telefone(re.sub(r"\D", "", numero)):
            termos.append((_digitos_com_separadores(variante), "TELEFONE", False))
    for email in re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", contato):
        termos.append((_palavra(normalizar(email)), "EMAIL", False))
        termos.append((_email_falado(email), "EMAIL", False))

    email = (dados_paciente.get("email") or "").strip()
    if email:
        termos.append((_palavra(normalizar(email)), "EMAIL", False))
        termos.append((_email_falado(email), "EMAIL", False))

    for variante in _variantes_de_telefone(re.sub(r"\D", "", dados_paciente.get("telefone") or "")):
        termos.append((_digitos_com_separadores(variante), "TELEFONE", False))

    cpf = re.sub(r"\D", "", dados_paciente.get("cpf") or "")
    if len(cpf) == 11:
        termos.append((_digitos_com_separadores(cpf), "CPF", False))

    return [(re.compile(padrao), tipo, exige_maiuscula) for padrao, tipo, exige_maiuscula in termos]


class DenyListPacienteRecognizer(EntityRecognizer):
    """
    Camada 1. Criado a cada requisição (vai como `ad_hoc_recognizers`), porque
    depende do paciente da sessão. A comparação é feita numa cópia normalizada do
    texto, que tem as mesmas posições do original.
    """

    def __init__(self, dados_paciente: dict, lexico=None):
        self._termos = montar_termos(dados_paciente, lexico)
        super().__init__(
            supported_entities=["PACIENTE", "PESSOA", "CPF", "TELEFONE", "EMAIL"],
            name="DenyListPaciente",
            supported_language=IDIOMA,
        )

    def load(self) -> None:
        pass

    def analyze(self, text, entities, nlp_artifacts=None):
        if not self._termos:
            return []
        normalizado = normalizar(text)
        return [
            _resultado(self, tipo, encontrado.start(), encontrado.end(), 1.0)
            for padrao, tipo, exige_maiuscula in self._termos
            for encontrado in padrao.finditer(normalizado)
            # A cópia normalizada é minúscula; a maiúscula se confere no original.
            if not exige_maiuscula or text[encontrado.start()].isupper()
        ]


# --- Camada 2: padrões brasileiros ---

class CpfRecognizer(PatternRecognizer):
    """
    CPF com ou sem pontuação. Dígito verificador certo → score máximo.
    Formatado (000.000.000-00) com dígito errado continua censurado, com score
    menor — um CPF ditado com um número trocado ainda identifica alguém. Já 11
    dígitos soltos só contam se o verificador bater: sem a pontuação, poderiam ser
    qualquer número.
    """

    def __init__(self):
        super().__init__(
            supported_entity="CPF",
            name="Cpf",
            supported_language=IDIOMA,
            patterns=[Pattern("cpf", r"(?<!\d)\d{3}[.\s]?\d{3}[.\s]?\d{3}[\s.-]?\d{2}(?!\d)", 0.6)],
            global_regex_flags=FLAGS,
        )

    def validate_result(self, pattern_text: str):
        if cpf_valido(pattern_text):
            return True
        formatado = not pattern_text.isdigit()
        # None mantém o score do padrão; False descarta o resultado.
        return None if formatado else False


def cpf_valido(texto: str) -> bool:
    """Confere os dois dígitos verificadores, ignorando a pontuação."""
    digitos = re.sub(r"\D", "", texto)
    if len(digitos) != 11 or digitos == digitos[0] * 11:
        return False
    # Cada dígito verificador é o resto da soma ponderada dos anteriores.
    for quantidade in (9, 10):
        soma = sum(int(digitos[i]) * (quantidade + 1 - i) for i in range(quantidade))
        resto = (soma * 10) % 11
        if (0 if resto == 10 else resto) != int(digitos[quantidade]):
            return False
    return True


class TelefoneRecognizer(PatternRecognizer):
    """Fixo e celular, com ou sem DDD e +55."""

    PADROES = [
        Pattern(
            "telefone_com_ddd",
            r"(?<![\w+])(?:\+?\s?55[\s.-]?)?(?:\(\s?\d{2}\s?\)|\d{2})[\s.-]?(?:9[\s.-]?)?\d{4}[\s.-]?\d{4}(?!\d)",
            0.7,
        ),
        # Sem DDD, exige o separador no meio — senão qualquer número de 8 dígitos
        # (protocolo, CEP sem traço) viraria telefone.
        Pattern("telefone_local", r"(?<![\w+])(?:9[\s.-]?)?\d{4}[\s.-]\d{4}(?!\d)", 0.6),
        Pattern("celular_sem_separador", r"(?<![\w+])9\d{8}(?!\d)", 0.6),
    ]

    def __init__(self):
        super().__init__(
            supported_entity="TELEFONE",
            name="Telefone",
            supported_language=IDIOMA,
            patterns=list(self.PADROES),
            global_regex_flags=FLAGS,
        )

    def invalidate_result(self, pattern_text: str):
        # "de 2019 2020": dois anos lado a lado não são telefone.
        grupos = re.findall(r"\d{4}", pattern_text)
        return (
            len(re.sub(r"\D", "", pattern_text)) == 8
            and all(1900 <= int(grupo) <= 2099 for grupo in grupos)
        )


class CepRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(
            supported_entity="CEP",
            name="Cep",
            supported_language=IDIOMA,
            patterns=[
                Pattern("cep", r"(?<!\d)\d{5}-\d{3}(?!\d)", 0.8),
                Pattern("cep_pontuado", r"(?<!\d)\d{2}\.\d{3}-\d{3}(?!\d)", 0.8),
                # "CEP 01310100": sem o traço, só com a palavra antes.
                Pattern("cep_rotulado", r"(?<=\bcep\W{0,3})\d{5}\.?-?\d{3}(?!\d)", 0.9),
            ],
            global_regex_flags=FLAGS,
        )


class EmailRecognizer(PatternRecognizer):
    def __init__(self):
        super().__init__(
            supported_entity="EMAIL",
            name="Email",
            supported_language=IDIOMA,
            patterns=[Pattern("email", r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", 1.0)],
            global_regex_flags=FLAGS,
        )


class EmailFaladoRecognizer(PatternRecognizer):
    """A transcrição escreve o e-mail como foi dito: 'joana arroba gmail ponto com'."""

    def __init__(self):
        super().__init__(
            supported_entity="EMAIL",
            name="EmailFalado",
            supported_language=IDIOMA,
            patterns=[
                Pattern(
                    "email_falado",
                    r"(?<!\w)\w+(?:\s+(?:ponto|underline|tra[cç]o|tracinho|h[ií]fen)\s+\w+)*"
                    r"\s+arroba\s+\w+(?:\s+ponto\s+\w+)+",
                    0.9,
                )
            ],
            global_regex_flags=FLAGS,
        )


_MESES = "janeiro|fevereiro|mar[cç]o|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro"


class DataRecognizer(PatternRecognizer):
    """Datas completas. 'Na terça' ou 'em março' não identificam ninguém e ficam."""

    def __init__(self):
        super().__init__(
            supported_entity="DATA",
            name="Data",
            supported_language=IDIOMA,
            patterns=[
                Pattern(
                    "data_numerica",
                    r"(?<![\d/.-])(?:0?[1-9]|[12]\d|3[01])([/.-])(?:0?[1-9]|1[0-2])\1(?:\d{4}|\d{2})(?![\d/.-]?\d)",
                    0.9,
                ),
                Pattern(
                    "data_por_extenso",
                    rf"(?<!\d)(?:0?[1-9]|[12]\d|3[01])(?:º|°)?\s+de\s+(?:{_MESES})\s+de\s+\d{{4}}(?!\d)",
                    0.8,
                ),
            ],
            global_regex_flags=FLAGS,
        )


def reconhecedores_de_padrao() -> list[PatternRecognizer]:
    return [
        CpfRecognizer(),
        TelefoneRecognizer(),
        CepRecognizer(),
        EmailRecognizer(),
        EmailFaladoRecognizer(),
        DataRecognizer(),
    ]


# --- Camada 3: dicionário de nomes do IBGE ---

def _ler_lista(nome_arquivo: str) -> frozenset[str]:
    linhas = (PASTA_DADOS / nome_arquivo).read_text(encoding="utf-8").splitlines()
    return frozenset(
        normalizar(linha.strip()) for linha in linhas if linha.strip() and not linha.startswith("#")
    )


@lru_cache(maxsize=1)
def nomes_ibge() -> frozenset[str]:
    return _ler_lista("nomes_ibge.txt")


@lru_cache(maxsize=1)
def nomes_ambiguos() -> frozenset[str]:
    return _ler_lista("nomes_ambiguos.txt")


@lru_cache(maxsize=1)
def termos_liberados() -> frozenset[str]:
    """Marcas, rótulos de fala etc. que nunca identificam ninguém (camadas 3 e 4)."""
    return _ler_lista("termos_liberados.txt")


CONECTIVOS_DE_NOME = {"de", "da", "do", "das", "dos"}
# Pronomes de tratamento ficam fora do trecho: "Dona [PESSOA_1]" é mais útil para
# o insight que "[PESSOA_1]", e o tratamento sozinho não identifica ninguém.
TRATAMENTOS = {
    "dona", "dom", "seu", "sr", "sra", "srta", "dr", "dra", "doutor", "doutora",
    "professor", "professora", "prof", "profa", "tia", "tio", "vo", "vovo", "padre",
    "pastor", "pastora", "irma", "irmao", "santo", "santa", "sao", "frei", "madre",
}
# O que encerra uma frase: depois disso, a transcrição capitaliza a palavra seguinte.
FIM_DE_FRASE = {".", "!", "?", "…", "...", ":", ";"}


def _capitalizado(token) -> bool:
    return token.is_alpha and len(token.text) >= 2 and token.text[0].isupper()


def _inicio_de_frase(doc, indice: int) -> bool:
    anterior = indice - 1
    while anterior >= 0 and (doc[anterior].is_space or doc[anterior].text in {'"', "“", "'", "‘", "-", "—", "–", "("}):
        if "\n" in doc[anterior].text:
            return True
        anterior -= 1
    return anterior < 0 or doc[anterior].text in FIM_DE_FRASE or "\n" in doc[anterior].whitespace_


class NomesIbgeRecognizer(EntityRecognizer):
    """
    Camada 3. Marca palavra capitalizada que está na lista de nomes, estendendo
    o trecho pelos sobrenomes (palavras capitalizadas seguidas, com "de/da/dos" no
    meio). Dois cuidados contra falso positivo:

    - nomes de `nomes_ambiguos.txt` (Rosa, Paz, Luz...) só contam acompanhados de
      outro nome ou sobrenome;
    - no início de frase a transcrição capitaliza tudo: lá, uma palavra solta só
      conta se estiver na lista (não é estendida para trás).
    """

    def __init__(self, nomes: frozenset[str] | None = None, ambiguos: frozenset[str] | None = None):
        self._nomes = nomes if nomes is not None else nomes_ibge()
        self._ambiguos = ambiguos if ambiguos is not None else nomes_ambiguos()
        super().__init__(supported_entities=["PESSOA"], name="NomesIbge", supported_language=IDIOMA)

    def load(self) -> None:
        pass

    def _eh_nome(self, token) -> bool:
        return _capitalizado(token) and normalizar(token.text) in self._nomes

    def analyze(self, text, entities, nlp_artifacts=None):
        if nlp_artifacts is None or nlp_artifacts.tokens is None:
            return []
        doc = nlp_artifacts.tokens
        resultados = []
        indice = 0
        while indice < len(doc):
            if not self._eh_nome(doc[indice]):
                indice += 1
                continue

            primeiro, ultimo = self._estender(doc, indice)
            acompanhado = ultimo > primeiro
            if normalizar(doc[indice].text) in self._ambiguos and not acompanhado:
                indice += 1
                continue

            if acompanhado:
                score = 0.85
            elif _inicio_de_frase(doc, indice):
                score = 0.6
            else:
                score = 0.8
            inicio = doc[primeiro].idx
            fim = doc[ultimo].idx + len(doc[ultimo].text)
            if chave(text[inicio:fim]) not in termos_liberados():
                resultados.append(_resultado(self, "PESSOA", inicio, fim, score))
            indice = ultimo + 1
        return resultados

    def _estender(self, doc, indice: int) -> tuple[int, int]:
        """Primeiro e último token do nome completo em volta de `doc[indice]`."""
        ultimo = indice
        proximo = indice + 1
        while proximo < len(doc) and doc[proximo - 1].whitespace_ == " ":
            token = doc[proximo]
            if _capitalizado(token):
                ultimo = proximo
                proximo += 1
            elif (
                token.lower_ in CONECTIVOS_DE_NOME
                and token.whitespace_ == " "
                and proximo + 1 < len(doc)
                and _capitalizado(doc[proximo + 1])
            ):
                ultimo = proximo + 1
                proximo += 2
            else:
                break

        # Para trás: sobrenome antes do nome ("Silva Rosa"), mas sem pronome de
        # tratamento e sem palavra comum capitalizada só por abrir a frase.
        primeiro = indice
        while primeiro > 0 and doc[primeiro - 1].whitespace_ == " ":
            token = doc[primeiro - 1]
            if not _capitalizado(token) or normalizar(token.text) in TRATAMENTOS:
                break
            if _inicio_de_frase(doc, primeiro - 1) and not self._eh_nome(token):
                break
            primeiro -= 1
        return primeiro, ultimo


# --- Camada 4: NER do spaCy, filtrado ---

# Palavra com razão maiúscula/minúscula a partir daqui é "comum" (ver lexico.py).
# Para o que está na lista de nomes do IBGE o limiar é bem maior: "Vitória" (2,3)
# continua sendo nome, "Irmã" (6,9) e "Domingo" (6,6) não.
LIMIAR_NOME_COMUM = 5.0
# No meio da frase a maiúscula já é um sinal de nome próprio ("em Contagem", 2,1):
# só sai o que é muito comum ("valor Total", 13; "minha Mãe", 8,4).
LIMIAR_MEIO_DE_FRASE = 5.0


class NerSpacyRecognizer(SpacyRecognizer):
    """
    Camada 4. O NER do spaCy português marca quase tudo que começa com maiúscula
    — "Total.", "Ansiedade. Medo.", "Saudade da minha mãe" viravam [PESSOA_n], e
    "Instagram" também. Os resultados dele passam por estes filtros, em ordem:

    1. termo liberado (`dados/termos_liberados.txt`) → descartado;
    2. pronome de tratamento na frente ("Dona", "Doutor") → cortado do trecho; e
       se o trecho começa com maiúscula e tem palavra minúscula que não é
       conectivo ("Saudade da minha"), é cortado antes dela;
    3. no início de frase ou de linha, onde a maiúscula não informa nada: se todas
       as palavras são comuns pelo léxico → descartado;
    4. no meio da frase, palavra única fora da lista de nomes e muito comum →
       descartada. Trechos de várias palavras ("Hospital das Clínicas") ficam;
    5. o que o NER marcou em minúsculas ("lucas") fica: é sinal forte de nome.

    Em caso de dúvida, censura: o que sobra de mais fica visível na revisão; o que
    escapa, não.
    """

    def __init__(self, lexico: Lexico, nomes: frozenset[str], liberados: frozenset[str], **opcoes):
        self._lexico = lexico
        self._nomes = nomes
        self._liberados = liberados
        super().__init__(**opcoes)

    def analyze(self, text, entities, nlp_artifacts=None):
        resultados = super().analyze(text, entities, nlp_artifacts)
        if not resultados or nlp_artifacts is None or nlp_artifacts.tokens is None:
            return resultados
        doc = nlp_artifacts.tokens

        filtrados = []
        for resultado in resultados:
            if chave(text[resultado.start:resultado.end]) in self._liberados:
                continue
            trecho = doc.char_span(resultado.start, resultado.end, alignment_mode="expand")
            if trecho is None:
                filtrados.append(resultado)
                continue

            tokens = list(trecho)
            # Só em pessoa: em lugar, "São" e "Santa" fazem parte do nome ("São Paulo").
            while tokens and (
                not tokens[0].is_alpha
                or (resultado.entity_type == "PESSOA" and normalizar(tokens[0].text).rstrip(".") in TRATAMENTOS)
            ):
                tokens = tokens[1:]
            # Nome não tem palavra minúscula no meio, fora os conectivos: quando o
            # NER engole o resto da frase ("Saudade da minha"), o trecho é cortado
            # na primeira e perde os conectivos que sobrarem na ponta.
            if tokens and tokens[0].text[:1].isupper():
                for posicao, token in enumerate(tokens):
                    if token.is_alpha and token.text[0].islower() and token.lower_ not in CONECTIVOS_DE_NOME:
                        tokens = tokens[:posicao]
                        break
                while tokens and (not tokens[-1].is_alpha or tokens[-1].lower_ in CONECTIVOS_DE_NOME):
                    tokens = tokens[:-1]
            if not tokens:
                continue
            resultado.start = tokens[0].idx
            resultado.end = tokens[-1].idx + len(tokens[-1].text)

            if chave(text[resultado.start:resultado.end]) in self._liberados:
                continue
            if self._palavras_comuns(doc, tokens):
                continue
            filtrados.append(resultado)
        return filtrados

    def _palavras_comuns(self, doc, tokens) -> bool:
        palavras = [t for t in tokens if t.is_alpha and t.lower_ not in CONECTIVOS_DE_NOME]
        if not palavras or not all(t.text[0].isupper() for t in palavras):
            return False

        if _inicio_de_frase(doc, tokens[0].i):
            return all(self._lexico.eh_palavra_comum(t.text, self._limiar(t)) for t in palavras)

        if len(tokens) == 1:
            palavra = palavras[0]
            return normalizar(palavra.text) not in self._nomes and self._lexico.eh_palavra_comum(
                palavra.text, LIMIAR_MEIO_DE_FRASE
            )
        return False

    def _limiar(self, token) -> float:
        return LIMIAR_NOME_COMUM if normalizar(token.text) in self._nomes else LIMIAR_PALAVRA_COMUM
