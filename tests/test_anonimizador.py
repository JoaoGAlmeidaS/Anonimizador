"""
Anonimizador (censura local).

Os testes rápidos usam um pipeline spaCy em branco — sem NER, só as camadas 1 a 3 —
e rodam sem modelo baixado. Os que precisam do modelo de verdade têm o marcador
`modelo` e ficam fora do `pytest` padrão:

    ANONIMIZADOR_MODELO=pt_core_news_md python -m pytest -m modelo -s

(`ANONIMIZADOR_MODELO` aceita o nome do pacote spaCy ou o caminho da pasta.)
"""
import logging
import os
import time
import warnings

import pytest
import spacy

from anonimizador import anonimizar, reverter
from anonimizador.lexico import Lexico
from anonimizador.marcadores import Trecho, resolver_sobreposicoes
from anonimizador.motor import VARIAVEL_MODELO, MotorCensura, carregar_spacy
from tests.avaliacao import avaliar, carregar_fixtures, recall_por_camada

FIXTURES = carregar_fixtures()


@pytest.fixture(scope="module")
def motor():
    """Camadas 1 a 3: o pipeline em branco tokeniza, mas não tem NER."""
    return MotorCensura(spacy.blank("pt"))


def censurar(motor, texto, **paciente):
    return anonimizar(texto, {"nome": "Paciente Ficticio", **paciente}, motor=motor)


# --- Configuração do Presidio ---

def test_so_os_reconhecedores_em_portugues(motor):
    # Registro vazio faria o Presidio carregar os pré-definidos em inglês — e os de
    # e-mail/URL usam o tldextract, que busca a lista de sufixos na internet.
    assert motor.nomes_dos_reconhecedores() == {
        "Cpf", "Telefone", "Cep", "Email", "EmailFalado", "Data", "NomesIbge", "NerSpacy",
    }


def test_ha_pelo_menos_vinte_fixtures():
    assert len(FIXTURES) >= 20


# --- Critérios de aceite sobre os fixtures ---

def test_recall_de_100_por_cento_na_deny_list_e_no_regex(motor):
    escaparam = []
    for fixture in FIXTURES:
        resultado = anonimizar(fixture["texto"], fixture["paciente"], motor=motor)
        escaparam += [
            f"{linha['arquivo']}: {linha['trecho']!r} (camada {linha['camada']})"
            for linha in avaliar(fixture, resultado)
            if linha["camada"] in (1, 2) and not linha["coberto"]
        ]
    assert not escaparam, "escaparam da censura:\n" + "\n".join(escaparam)


def test_o_que_nao_identifica_ninguem_continua_no_texto(motor):
    sumiram = []
    for fixture in FIXTURES:
        censurado = anonimizar(fixture["texto"], fixture["paciente"], motor=motor).texto_censurado
        sumiram += [
            f"{fixture['arquivo']}: {trecho!r}"
            for trecho in fixture.get("preservar", [])
            if trecho not in censurado
        ]
    assert not sumiram, "censurados sem necessidade:\n" + "\n".join(sumiram)



def test_recall_da_camada_3_sem_ner(motor):
    """Medido e reportado (meta a definir com o usuário). Aqui só não pode regredir."""
    linhas = []
    for fixture in FIXTURES:
        resultado = anonimizar(fixture["texto"], fixture["paciente"], motor=motor)
        linhas += avaliar(fixture, resultado)
    cobertas, total = recall_por_camada(linhas)[3]
    assert cobertas == total


# --- Marcadores ---

def test_mesmo_valor_mesmo_marcador_e_numeracao_por_ordem(motor):
    fixture = next(f for f in FIXTURES if f["arquivo"].startswith("18_"))
    resultado = anonimizar(fixture["texto"], fixture["paciente"], motor=motor)

    assert resultado.texto_censurado.count("[PESSOA_1]") == 2  # Raimundo
    assert resultado.mapa["[PESSOA_1]"] == "Raimundo"
    assert resultado.mapa["[PESSOA_5]"] == "Raimunda"
    assert resultado.texto_censurado.startswith("[PACIENTE] falou")
    assert resultado.contagem == {"PACIENTE": 1, "PESSOA": 5}


def test_mesma_pessoa_com_caixa_diferente_e_o_mesmo_marcador(motor):
    resultado = censurar(motor, "O Carlos ligou. Depois o CARLOS ligou de novo.")
    assert resultado.texto_censurado == "O [PESSOA_1] ligou. Depois o [PESSOA_1] ligou de novo."


def test_terceiro_com_o_sobrenome_do_paciente_e_pessoa(motor):
    resultado = anonimizar(
        "O Carlos Ferraz e a família Ferraz.", {"nome": "Natália Ferraz"}, motor=motor
    )
    assert resultado.texto_censurado == "O [PESSOA_1] e a família [PACIENTE]."


def test_tratamento_antes_do_nome_fica(motor):
    resultado = anonimizar(
        "A Dona Ana Souza veio.", {"nome": "Ana Souza"}, motor=motor
    )
    assert resultado.texto_censurado == "A Dona [PACIENTE] veio."


def test_email_do_paciente_com_o_nome_dentro_e_email(motor):
    resultado = anonimizar(
        "Escreva para rafaela.torres@exemplo.net hoje.",
        {"nome": "Rafaela Torres", "email": "rafaela.torres@exemplo.net"},
        motor=motor,
    )
    assert resultado.texto_censurado == "Escreva para [EMAIL_1] hoje."


def test_segmentos_remontam_o_original_e_o_censurado(motor):
    for fixture in FIXTURES:
        resultado = anonimizar(fixture["texto"], fixture["paciente"], motor=motor)
        segmentos = resultado.segmentos
        assert "".join(s["texto"] for s in segmentos) == fixture["texto"]
        assert "".join(s["marcador"] or s["texto"] for s in segmentos) == resultado.texto_censurado


# --- Camada 2: padrões ---

@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("CPF 418.257.306-44.", "CPF [CPF_1]."),
        ("CPF 41825730644.", "CPF [CPF_1]."),
        # Formatado com dígito errado: censurado mesmo assim.
        ("CPF 123.456.789-00.", "CPF [CPF_1]."),
        # 11 dígitos soltos com verificador errado: pode ser qualquer número.
        ("Protocolo 12345678900.", "Protocolo 12345678900."),
    ],
)
def test_cpf(motor, texto, esperado):
    assert censurar(motor, texto).texto_censurado == esperado


@pytest.mark.parametrize(
    "texto",
    [
        "Liga no (11) 98765-4321.",
        "Liga no 11987654321.",
        "Liga no +55 11 98765-4321.",
        "Liga no 3333-4444.",
        "Liga no 98765-4321.",
        "Liga no 987654321.",
    ],
)
def test_telefones(motor, texto):
    assert censurar(motor, texto).texto_censurado == "Liga no [TELEFONE_1]."


def test_dois_anos_lado_a_lado_nao_sao_telefone(motor):
    assert censurar(motor, "Foi entre 2019 2020.").texto_censurado == "Foi entre 2019 2020."


@pytest.mark.parametrize("cep", ["01310-100", "01.310-100", "CEP 01310100"])
def test_cep(motor, cep):
    assert "[CEP_1]" in censurar(motor, f"Moro no {cep}.").texto_censurado


def test_email_escrito_e_falado(motor):
    resultado = censurar(motor, "Mande para joana@exemplo.com ou joana ponto lima arroba exemplo ponto com.")
    assert resultado.texto_censurado == "Mande para [EMAIL_1] ou [EMAIL_2]."


@pytest.mark.parametrize("data", ["14/02/1990", "14/2/90", "14.02.1990", "3 de setembro de 2015"])
def test_data_completa(motor, data):
    assert censurar(motor, f"Foi em {data}.").texto_censurado == "Foi em [DATA_1]."


def test_data_parcial_fica(motor):
    texto = "Em março, na terça, e no dia 12 de maio."
    assert censurar(motor, texto).texto_censurado == texto


# --- Camada 1: deny-list ---

def test_deny_list_ignora_caixa_e_acento(motor):
    resultado = anonimizar("o JOAO e o joão.", {"nome": "João Silva"}, motor=motor)
    assert resultado.texto_censurado == "o [PACIENTE] e o [PACIENTE]."


def test_partes_curtas_e_preposicoes_nao_entram_na_deny_list(motor):
    resultado = anonimizar("Lu e da casa de Lu.", {"nome": "Lu da Costa"}, motor=motor)
    assert resultado.texto_censurado == "Lu e da casa de Lu."


def test_telefone_do_paciente_ditado_digito_a_digito(motor):
    resultado = anonimizar(
        "o número é 9 8 7 6 5 4 3 2 1", {"nome": "X", "telefone": "(11) 98765-4321"}, motor=motor
    )
    assert resultado.texto_censurado == "o número é [TELEFONE_1]"


def test_contato_de_emergencia_em_texto_livre(motor):
    resultado = anonimizar(
        "A Joana ligou do 91234-5678.",
        {"nome": "Ana Souza", "contato_emergencia": "Joana Lima (mãe) 11 91234-5678"},
        motor=motor,
    )
    assert resultado.texto_censurado == "A [PESSOA_1] ligou do [TELEFONE_1]."


# --- Camada 3: nomes do IBGE ---

def test_nome_ambiguo_sozinho_fica_e_acompanhado_entra(motor):
    assert censurar(motor, "A Rosa ligou.").texto_censurado == "A Rosa ligou."
    assert censurar(motor, "A Rosa Campos ligou.").texto_censurado == "A [PESSOA_1] ligou."


def test_marco_sem_acento_no_inicio_da_frase_nao_vira_nome(motor):
    texto = "Março foi difícil. Domingo também."
    assert censurar(motor, texto).texto_censurado == texto


def test_nome_no_inicio_da_frase_conta(motor):
    assert censurar(motor, "Paula chegou.").texto_censurado == "[PESSOA_1] chegou."


def test_nome_estendido_por_sobrenome_e_conectivo(motor):
    resultado = censurar(motor, "A avó, Maria das Dores Silva, faleceu.")
    assert resultado.texto_censurado == "A avó, [PESSOA_1], faleceu."


# --- Camada 4 filtrada: palavra comum com maiúscula não é nome ---
#
# O NER de verdade só existe com o modelo baixado. Aqui ele é imitado por um
# `entity_ruler` que rotula exatamente o que o spaCy pt rotulava errado, e o léxico
# recebe as posições medidas nos vetores do modelo (ver lexico.py).

POSICOES = {
    "Total": 5821, "total": 440,
    "Ansiedade": 33179, "ansiedade": 3567,
    "Saudade": 19468, "saudade": 5998,
    "Mãe": 4088, "mãe": 484,
    "Juliana": 7294, "juliana": 59329,
    "Contagem": 15028, "contagem": 7078,
    "Vitória": 2881, "vitória": 1237,
    "Irmã": 16780, "irmã": 2442,
    "Instagram": 3264, "instagram": 10062,
    "Leite": 4435, "leite": 1513,
}


def motor_com_ner_falso(rotulos: dict[str, str]) -> MotorCensura:
    nlp = spacy.blank("pt")
    if rotulos:
        regua = nlp.add_pipe("entity_ruler")
        regua.add_patterns([{"label": rotulo, "pattern": texto} for texto, rotulo in rotulos.items()])
    return MotorCensura(nlp, lexico=Lexico.de_posicoes(POSICOES))


def censurar_com_ner(rotulos, texto, **paciente):
    return anonimizar(texto, {"nome": "Paciente Ficticio", **paciente}, motor=motor_com_ner_falso(rotulos))


@pytest.mark.parametrize(
    "texto",
    [
        "Total.\nFoi uma semana pesada.",       # início de linha (transcrição)
        "Ansiedade. Medo. Tristeza.",           # início de frase
        "Ela disse: Ansiedade demais.",          # depois de dois-pontos
        "Ela disse que o valor Total subiu.",    # meio da frase, palavra muito comum
        "A minha Mãe ligou.",
    ],
)
def test_palavra_comum_com_maiuscula_nao_vira_pessoa(texto):
    rotulos = {"Total": "PER", "Ansiedade": "PER", "Mãe": "PER"}
    assert censurar_com_ner(rotulos, texto).texto_censurado == texto


def test_nome_continua_censurado_no_inicio_da_frase():
    assert censurar_com_ner({"Juliana": "PER"}, "Juliana ligou.").texto_censurado == "[PESSOA_1] ligou."


def test_nome_ambiguo_decide_pela_frequencia():
    # Vitória (2,3) é usado como nome; Irmã (6,9) é quase sempre a palavra.
    rotulos = {"Vitória": "PER", "Irmã": "PER"}
    assert censurar_com_ner(rotulos, "Vitória ligou.").texto_censurado == "[PESSOA_1] ligou."
    assert censurar_com_ner(rotulos, "Irmã dela veio.").texto_censurado == "Irmã dela veio."


def test_maiuscula_no_meio_da_frase_vale_para_o_que_nao_e_muito_comum():
    # "contagem" é palavra, mas no meio da frase com maiúscula é a cidade.
    rotulos = {"Contagem": "LOC"}
    assert censurar_com_ner(rotulos, "Ela mora em Contagem.").texto_censurado == "Ela mora em [LOCAL_1]."
    # No início da frase a maiúscula não informa nada: fica (troca documentada).
    assert censurar_com_ner(rotulos, "Contagem é longe.").texto_censurado == "Contagem é longe."


def test_trecho_de_varias_palavras_no_meio_da_frase_fica():
    rotulos = {"Hospital das Clínicas": "ORG"}
    resultado = censurar_com_ner(rotulos, "Fica perto do Hospital das Clínicas.")
    assert resultado.texto_censurado == "Fica perto do [ORGANIZACAO_1]."


def test_trecho_do_ner_que_engole_a_frase_e_cortado():
    # O spaCy lg devolveu "Saudade da minha" como uma pessoa só.
    rotulos = {"Saudade da minha": "PER", "Pedro meu": "PER"}
    assert censurar_com_ner(rotulos, "Saudade da minha mãe.").texto_censurado == "Saudade da minha mãe."
    assert censurar_com_ner(rotulos, "Falei com o Pedro meu irmão.").texto_censurado == "Falei com o [PESSOA_1] meu irmão."


def test_nome_em_minusculas_marcado_pelo_ner_fica():
    rotulos = {"lucas": "PER"}
    assert censurar_com_ner(rotulos, "falei com o lucas ontem").texto_censurado == "falei com o [PESSOA_1] ontem"


@pytest.mark.parametrize("texto", ["Ela me mandou no Instagram.", "Instagram é onde ela posta tudo."])
def test_rede_social_e_termo_liberado(texto):
    assert censurar_com_ner({"Instagram": "PER"}, texto).texto_censurado == texto


def test_tratamento_sai_do_trecho_da_pessoa_mas_nao_do_lugar():
    rotulos = {"Dona Conceição Ribeiro": "PER", "São Paulo": "LOC"}
    resultado = censurar_com_ner(rotulos, "A Dona Conceição Ribeiro mora em São Paulo.")
    assert resultado.texto_censurado == "A Dona [PESSOA_1] mora em [LOCAL_1]."


def test_sem_lexico_nada_deixa_de_ser_censurado():
    # Pipeline sem vetores: o léxico não opina e o NER vale como antes.
    nlp = spacy.blank("pt")
    nlp.add_pipe("entity_ruler").add_patterns([{"label": "PER", "pattern": "Total"}])
    resultado = anonimizar("Total.", {"nome": "X"}, motor=MotorCensura(nlp))
    assert resultado.texto_censurado == "[PESSOA_1]."


def test_rotulo_ruidoso_do_ner_perde_para_o_dicionario_de_nomes():
    # O spaCy md chama "Juliana" de lugar; o dicionário do IBGE sabe que é pessoa.
    resultado = censurar_com_ner({"Juliana": "LOC"}, "Falei com a Juliana ontem.")
    assert resultado.texto_censurado == "Falei com a [PESSOA_1] ontem."


# --- Deny-list: parte do nome que é palavra comum ---

def test_sobrenome_que_e_palavra_comum_so_conta_com_maiuscula():
    motor = motor_com_ner_falso({})
    resultado = anonimizar(
        "Tomei leite. A Leite não veio. carolina leite chegou.", {"nome": "Carolina Leite"}, motor=motor
    )
    assert resultado.texto_censurado == "Tomei leite. A [PACIENTE] não veio. [PACIENTE] chegou."


def test_par_de_nomes_pulando_o_do_meio_vale_em_minusculas():
    motor = motor_com_ner_falso({})
    resultado = anonimizar("o joão leite ligou", {"nome": "João Pedro Leite"}, motor=motor)
    assert resultado.texto_censurado == "o [PACIENTE] ligou"


def test_nome_ambiguo_do_paciente_sozinho_precisa_de_maiuscula(motor):
    # "Filho" está em nomes_ambiguos.txt: funciona mesmo sem o léxico do modelo.
    resultado = anonimizar("Meu filho falou com o Pedro Filho.", {"nome": "Pedro Filho"}, motor=motor)
    assert resultado.texto_censurado == "Meu filho falou com o [PACIENTE]."


# --- Léxico tirado dos vetores do spaCy ---

def test_lexico_le_a_ordem_de_frequencia_dos_vetores():
    import numpy

    nlp = spacy.blank("pt")
    nlp.vocab.reset_vectors(width=2)
    for palavra in ("total", "Juliana", "Total", "juliana"):  # ordem = frequência
        nlp.vocab.set_vector(palavra, numpy.zeros(2, dtype="float32"))
    lexico = Lexico.do_spacy(nlp)

    assert lexico.razao("Total") == 2.0          # posição 2 / posição 0 (vira 1)
    assert lexico.razao("Juliana") == 1 / 3
    assert lexico.razao("Kethellyn") is None    # rara: não opina
    assert lexico.eh_palavra_comum("TOTAL", 1.5)
    assert not lexico.eh_palavra_comum("Juliana", 1.5)
    assert Lexico.do_spacy(spacy.blank("pt")).razao("Total") is None


# --- Sobreposições ---

def test_sobreposicao_vira_a_uniao_com_o_tipo_do_maior():
    trechos = resolver_sobreposicoes(
        [Trecho(0, 6, "PESSOA", 3, 0.8), Trecho(0, 15, "LOCAL", 4, 0.85)]
    )
    assert trechos == [Trecho(0, 15, "LOCAL", 3, 0.85)]


def test_paciente_vence_quando_cobre_metade_do_trecho():
    # "Dona Ana Souza" (NER) contém "Ana Souza" (deny-list): é o paciente.
    assert resolver_sobreposicoes(
        [Trecho(0, 14, "PESSOA", 4, 0.85), Trecho(5, 14, "PACIENTE", 1, 1.0)]
    )[0].tipo == "PACIENTE"
    # "Carlos Ferraz" (IBGE) contém só o sobrenome do paciente: é outra pessoa.
    assert resolver_sobreposicoes(
        [Trecho(0, 13, "PESSOA", 3, 0.85), Trecho(7, 13, "PACIENTE", 1, 1.0)]
    )[0].tipo == "PESSOA"


def test_trechos_que_so_se_encostam_nao_se_fundem():
    assert len(resolver_sobreposicoes([Trecho(0, 3, "PESSOA", 3, 1), Trecho(3, 6, "CPF", 2, 1)])) == 2


# --- Reversão ---

def test_reverter_aceita_as_variacoes_que_o_modelo_devolve():
    mapa = {"[PACIENTE]": "Ana", "[PESSOA_1]": "Carlos", "[LOCAL_1]": "Contagem"}
    texto = "[PACIENTE] e PESSOA_1 em [Local_1]; depois [pessoa 1] e Pessoa_1."
    assert reverter(texto, mapa) == "Ana e Carlos em Contagem; depois Carlos e Carlos."


def test_reverter_nao_toca_em_palavras_comuns():
    mapa = {"[PACIENTE]": "Ana", "[PESSOA_1]": "Carlos", "[DATA_1]": "12/03/1985"}
    texto = "O paciente falou de uma pessoa, de um local e de uma data."
    assert reverter(texto, mapa) == texto


def test_reverter_mantem_marcador_fora_do_mapa():
    assert reverter("[PESSOA_9] e [PESSOA_1]", {"[PESSOA_1]": "Carlos"}) == "[PESSOA_9] e Carlos"


def test_reverter_paciente_sem_colchete_so_em_caixa_alta():
    assert reverter("PACIENTE chegou; o paciente saiu.", {"[PACIENTE]": "Ana"}) == "Ana chegou; o paciente saiu."


def test_ida_e_volta_devolve_cada_marcador_na_primeira_grafia(motor):
    # Cada marcador volta com a primeira grafia que recebeu. Para pessoas é a mesma
    # pessoa (só a caixa pode mudar); [PACIENTE] é um marcador só para todas as
    # formas do nome ("Natália", "Ferraz", "Natália Ferraz") e volta na primeira.
    for fixture in FIXTURES:
        resultado = anonimizar(fixture["texto"], fixture["paciente"], motor=motor)
        esperado = "".join(
            resultado.mapa[s["marcador"]] if s["marcador"] else s["texto"]
            for s in resultado.segmentos
        )
        assert reverter(resultado.texto_censurado, resultado.mapa) == esperado
        for segmento in resultado.segmentos:
            if segmento["marcador"] and segmento["tipo"] != "PACIENTE":
                assert resultado.mapa[segmento["marcador"]].lower() == segmento["texto"].lower()


# --- Nada de texto clínico em log, saída ou aviso ---

def test_nenhum_texto_clinico_vaza_em_log_saida_ou_aviso(motor, caplog, capsys):
    caplog.set_level(logging.DEBUG)  # todos os loggers, inclusive os de terceiros
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        resultados = [
            anonimizar(fixture["texto"], fixture["paciente"], motor=motor) for fixture in FIXTURES
        ]
        print(resultados[0])  # o repr não pode trazer conteúdo

    saida = capsys.readouterr()
    vazamentos = caplog.text + saida.out + saida.err + " ".join(str(a.message) for a in avisos)

    sensiveis = set()
    for fixture in FIXTURES:
        sensiveis.update(esperado["trecho"] for esperado in fixture["esperado"])
        sensiveis.update(valor for valor in fixture["paciente"].values() if valor)
        sensiveis.update(frase.strip() for frase in fixture["texto"].split(".") if len(frase.strip()) > 20)
    vazados = sorted(trecho for trecho in sensiveis if len(trecho) >= 4 and trecho in vazamentos)
    assert not vazados, f"apareceram no log/saída: {vazados[:5]}"
    assert "censura:" in caplog.text  # o log de metadados existe


def test_repr_do_resultado_nao_mostra_conteudo(motor):
    resultado = censurar(motor, "O Carlos ligou do 98765-4321.")
    assert "Carlos" not in repr(resultado) and "98765" not in repr(resultado)


# --- Com o modelo spaCy de verdade (pytest -m modelo) ---

@pytest.fixture(scope="module")
def motor_com_modelo():
    modelo = os.getenv(VARIAVEL_MODELO)
    if not modelo:
        pytest.skip(f"defina {VARIAVEL_MODELO} com o nome ou a pasta do modelo spaCy")
    try:
        nlp = carregar_spacy(modelo)
    except OSError:
        pytest.skip(f"modelo spaCy {modelo!r} não encontrado")
    return MotorCensura(nlp)


@pytest.mark.modelo
def test_recall_com_ner(motor_com_modelo):
    linhas = []
    for fixture in FIXTURES:
        resultado = anonimizar(fixture["texto"], fixture["paciente"], motor=motor_com_modelo)
        linhas += avaliar(fixture, resultado)
    recall = recall_por_camada(linhas)
    for camada, (cobertas, total) in recall.items():
        print(f"camada {camada}: {cobertas}/{total}")
    assert recall[1][0] == recall[1][1] and recall[2][0] == recall[2][1]


@pytest.mark.modelo
def test_sem_falsos_positivos_com_ner(motor_com_modelo):
    """O NER de verdade marcava "Total", "Ansiedade", "Instagram"... como pessoa."""
    sumiram = []
    for fixture in FIXTURES:
        censurado = anonimizar(fixture["texto"], fixture["paciente"], motor=motor_com_modelo).texto_censurado
        sumiram += [
            f"{fixture['arquivo']}: {trecho!r}"
            for trecho in fixture.get("preservar", []) + fixture.get("preservar_com_ner", [])
            if trecho not in censurado
        ]
    assert not sumiram, "censurados sem necessidade:\n" + "\n".join(sumiram)


@pytest.mark.modelo
def test_tempo_de_uma_sessao_de_60_minutos(motor_com_modelo):
    # ~130 palavras por minuto de fala → ~8 mil palavras em 60 minutos.
    longo = next(f for f in FIXTURES if f["arquivo"].startswith("22_"))
    palavras_por_bloco = len(longo["texto"].split())
    texto = "\n".join([longo["texto"]] * (8000 // palavras_por_bloco + 1))

    inicio = time.perf_counter()
    anonimizar(texto, longo["paciente"], motor=motor_com_modelo)
    duracao = time.perf_counter() - inicio
    print(f"{len(texto.split())} palavras ({len(texto)} caracteres) em {duracao:.1f}s")
    assert duracao < 120
