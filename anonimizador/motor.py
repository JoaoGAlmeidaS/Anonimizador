"""
Configuração do Presidio em português.

Três escolhas que não são estilo:

- **O spaCy é carregado por nós** (pelo nome do pacote ou pelo caminho da pasta) e
  injetado no `SpacyNlpEngine`. O `load()` do Presidio, se não encontrasse o
  modelo, chamaria `spacy.cli.download` (rede!) e ainda tenta `import torch` para
  ver se há GPU.
- **O registro de reconhecedores nunca pode ficar vazio**: vazio, o
  `AnalyzerEngine` carrega sozinho os reconhecedores pré-definidos (inglês/EUA),
  e os de e-mail e URL usam o tldextract, que busca a lista de sufixos na internet.
  `test_anonimizador.py` confere o conjunto exato.
- **`log_decision_process=False`**: ligado, o Presidio registra os trechos
  encontrados — ou seja, o texto clínico — num logger próprio.
"""
import logging
import os
import threading

from anonimizador.lexico import Lexico
from anonimizador.marcadores import Trecho
from anonimizador.reconhecedores import (
    CAMADA_POR_RECONHECEDOR,
    FLAGS,
    IDIOMA,
    DenyListPacienteRecognizer,
    NerSpacyRecognizer,
    NomesIbgeRecognizer,
    nomes_ibge,
    reconhecedores_de_padrao,
    termos_liberados,
)

logger = logging.getLogger(__name__)

# Os logs de depuração do Presidio só têm nomes de padrão e tempos, mas o nível
# fica em WARNING para nenhuma versão futura passar a escrever o trecho.
logging.getLogger("presidio-analyzer").setLevel(logging.WARNING)

# Rótulos do spaCy português → tipo do marcador. MISC (obras, eventos, idiomas)
# fica desligado: raramente identifica alguém e erra muito.
MAPEAMENTO_NER = {"PER": "PESSOA", "LOC": "LOCAL", "ORG": "ORGANIZACAO"}

# Componentes do pipeline que a censura não usa. Só o NER (e o tok2vec dele)
# interessa; o parser sozinho é boa parte do tempo de processamento.
COMPONENTES_DISPENSADOS = ["parser", "lemmatizer", "morphologizer", "attribute_ruler", "senter"]


# Modelo usado por `obter_motor`: nome do pacote spaCy ou caminho da pasta.
VARIAVEL_MODELO = "ANONIMIZADOR_MODELO"
MODELO_PADRAO = "pt_core_news_md"


def carregar_spacy(modelo):
    """Pipeline spaCy (nome do pacote ou pasta do modelo), só com o necessário."""
    import spacy

    return spacy.load(str(modelo), exclude=COMPONENTES_DISPENSADOS)


class MotorCensura:
    """
    Envolve o `AnalyzerEngine`. O mesmo motor pode ser chamado de várias threads
    (um servidor, uma interface), então a análise passa por uma trava: um texto de
    cada vez.
    """

    def __init__(self, nlp, limiar_ner: float = 0.5, lexico: Lexico | None = None):
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NerModelConfiguration, SpacyNlpEngine

        # Frequência das palavras, tirada dos vetores do próprio modelo: separa
        # "Total" (palavra comum) de "Juliana" (nome). Ver lexico.py.
        self._lexico = lexico or Lexico.do_spacy(nlp)

        configuracao_ner = NerModelConfiguration(
            model_to_presidio_entity_mapping=dict(MAPEAMENTO_NER),
            labels_to_ignore=["MISC"],
            default_score=0.85,
        )
        motor_nlp = SpacyNlpEngine(
            models=[{"lang_code": IDIOMA, "model_name": "injetado"}],
            ner_model_configuration=configuracao_ner,
        )
        motor_nlp.nlp = {IDIOMA: nlp}

        ner = NerSpacyRecognizer(
            lexico=self._lexico,
            nomes=nomes_ibge(),
            liberados=termos_liberados(),
            supported_language=IDIOMA,
            supported_entities=list(MAPEAMENTO_NER.values()),
            name="NerSpacy",
        )
        # O spaCy não dá confiança por entidade: o Presidio atribui 0,85 a todas.
        # Por isso o limiar, na prática, liga ou desliga a camada de NER.
        ner.score_thresholds = {"default": float(limiar_ner)}

        registro = RecognizerRegistry(
            recognizers=[*reconhecedores_de_padrao(), NomesIbgeRecognizer(), ner],
            supported_languages=[IDIOMA],
            global_regex_flags=FLAGS,
        )
        self._analisador = AnalyzerEngine(
            registry=registro,
            nlp_engine=motor_nlp,
            supported_languages=[IDIOMA],
            log_decision_process=False,
        )
        self._trava = threading.Lock()

    def nomes_dos_reconhecedores(self) -> set[str]:
        return {reconhecedor.name for reconhecedor in self._analisador.registry.recognizers}

    def analisar(self, texto: str, dados_paciente: dict) -> list[Trecho]:
        """Trechos encontrados pelas quatro camadas, ainda com sobreposições."""
        deny_list = DenyListPacienteRecognizer(dados_paciente, lexico=self._lexico)
        with self._trava:
            resultados = self._analisador.analyze(
                text=texto, language=IDIOMA, ad_hoc_recognizers=[deny_list]
            )
        return [
            Trecho(
                inicio=resultado.start,
                fim=resultado.end,
                tipo=resultado.entity_type,
                camada=CAMADA_POR_RECONHECEDOR.get(
                    resultado.recognition_metadata.get("recognizer_name"), 4
                ),
                score=resultado.score,
            )
            for resultado in resultados
        ]


# --- Motor padrão (um só, carregado na primeira censura) ---

_motor: MotorCensura | None = None
_trava_carga = threading.Lock()


def obter_motor() -> MotorCensura:
    """
    Motor com o modelo spaCy de `ANONIMIZADOR_MODELO` (padrão: `pt_core_news_md`).
    Carregar leva alguns segundos e centenas de MB, então acontece uma vez e fica
    em memória.
    """
    global _motor
    with _trava_carga:
        if _motor is None:
            modelo = os.getenv(VARIAVEL_MODELO) or MODELO_PADRAO
            try:
                nlp = carregar_spacy(modelo)
            except OSError as erro:
                raise RuntimeError(
                    f"Modelo spaCy '{modelo}' não encontrado. Instale com "
                    f"`python -m spacy download {MODELO_PADRAO}` ou aponte "
                    f"{VARIAVEL_MODELO} para a pasta do modelo."
                ) from erro
            _motor = MotorCensura(nlp)
            logger.info("motor de censura carregado (modelo spaCy %s)", nlp.meta.get("name"))
        return _motor


def descartar_motor() -> None:
    """Solta o motor (troca de modelo, testes)."""
    global _motor
    with _trava_carga:
        _motor = None
