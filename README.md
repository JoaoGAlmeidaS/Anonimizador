# Anonimizador

Censura local de dados pessoais em **texto clínico em português**: troca nomes,
documentos, contatos, lugares e datas por marcadores (`[PACIENTE]`, `[PESSOA_1]`,
`[TELEFONE_1]`…) antes de o texto sair da máquina, por exemplo para ser analisado
por uma LLM na nuvem. Depois, `reverter` põe os valores de volta na resposta.

Roda 100% offline. Usa o [Presidio](https://github.com/microsoft/presidio) só para
a detecção e o [spaCy](https://spacy.io) em português. Os reconhecedores, a troca
pelos marcadores e o filtro de falsos positivos são próprios.

Nasceu dentro de um sistema de gestão para psicólogos, onde censura anotações e
transcrições de sessão.

```python
from anonimizador import anonimizar, reverter

paciente = {"nome": "Carolina Mendes Leite", "telefone": "(11) 98765-4321"}
texto = (
    "Total. A Carolina contou que brigou com a Juliana na escola.\n"
    "Tomou leite quente e viajou para Porto Alegre em 12/03/2024.\n"
    "Pediu retorno no 11 98765-4321 ou no carol.leite@exemplo.com."
)

resultado = anonimizar(texto, paciente)
print(resultado.texto_censurado)
```

```
Total. A [PACIENTE] contou que brigou com a [PESSOA_1] na escola.
Tomou leite quente e viajou para [LOCAL_1] em [DATA_1].
Pediu retorno no [TELEFONE_1] ou no [EMAIL_1].
```

O exemplo mostra o que é difícil no problema:
- "Total" e "leite" ficaram. O NER do spaCy marcava "Total" como pessoa, e "Leite"
  é sobrenome da paciente;
- "Carolina" virou `[PACIENTE]` e não `[PESSOA_n]`;
- o e-mail contém o sobrenome, mas continua sendo um e-mail.

```python
resultado.contagem  # {'PACIENTE': 1, 'PESSOA': 1, 'LOCAL': 1, 'DATA': 1, 'TELEFONE': 1, 'EMAIL': 1}

resposta_da_llm = "A briga com [PESSOA_1] parece pesar para [PACIENTE]."
reverter(resposta_da_llm, resultado.mapa)  # 'A briga com Juliana parece pesar para Carolina.'
```

## Instalação

Requer Python 3.10+ (testado no 3.14).

```bash
pip install -e .
python -m spacy download pt_core_news_md
```

**Modelo spaCy:**
- o padrão é o `pt_core_news_md`;
- outro modelo vai pela variável `ANONIMIZADOR_MODELO`: o nome do pacote
  (`pt_core_news_lg`) ou o caminho de uma pasta com o modelo;
- também dá para montar o motor à mão e passá-lo a `anonimizar(..., motor=...)`:

```python
from anonimizador.motor import MotorCensura, carregar_spacy

motor = MotorCensura(carregar_spacy("pt_core_news_lg"))
resultado = anonimizar(texto, paciente, motor=motor)
```

**Dados do paciente.** `dados_paciente` aceita `nome`, `cpf`, `telefone`, `email` e
`contato_emergencia`. Todos são opcionais.

**O resultado traz:**
- `texto_censurado`;
- `mapa` (marcador → valor original);
- `contagem` (marcadores distintos por tipo);
- `trechos` (posição, tipo e camada de cada detecção);
- `segmentos` (o original fatiado em pedaços comuns e censurados, pronto para
  destacar numa tela de revisão).

## Como funciona

### Quatro camadas

| Camada | O que pega | Como |
|---|---|---|
| 1. Deny-list do paciente | o nome dele (inteiro, pares de partes, partes sozinhas), CPF, telefone, e-mail, contato de emergência | montada a cada chamada, sem diferenciar maiúsculas e acentos |
| 2. Padrões brasileiros | CPF, telefone fixo e celular, CEP, e-mail (inclusive ditado: "joana arroba gmail ponto com"), datas | regex; CPF sem pontuação só conta se o dígito verificador bater |
| 3. Nomes do IBGE | ~1 000 nomes próprios do Censo 2010 | dicionário + regras de contexto |
| 4. NER do spaCy | pessoas, lugares e organizações que as outras não conhecem | `pt_core_news_md`/`lg`, filtrado (abaixo) |

### Detalhes da deny-list

- **Pares de partes.** As pessoas pulam o nome do meio: "joão leite" é "João Pedro
  Leite".
- **Parte sozinha que também é palavra comum** ("Leite", "Campos") ou nome ambíguo
  ("Filho", "Neto") só conta com maiúscula. Sem isso, "tomei leite" virava
  "tomei [PACIENTE]".

### Detalhes da camada 3

- **Nomes que também são palavras** ("Rosa", "Glória", "Marco", que é "março" sem
  acento) só contam acompanhados de outro nome ou de um sobrenome.
- **Termos liberados** (`dados/termos_liberados.txt`: "Deus", "Instagram",
  "WhatsApp", "SUS"…) nunca são censurados pelas camadas 3 e 4.

### O filtro do NER: frequência de palavras

O NER do spaCy em português marca como pessoa quase tudo que começa com maiúscula.
Numa transcrição, toda linha começa com maiúscula: "Total.", "Ansiedade. Medo.",
"Saudade da minha mãe" viravam `[PESSOA_n]`. O limiar de confiança não ajuda,
porque o Presidio dá 0,85 a toda entidade do spaCy.

A pista vem do próprio modelo:
- o vocabulário dos vetores do spaCy diferencia maiúsculas e está em ordem de
  frequência;
- palavra comum aparece muito mais em minúsculas: "total" é a 440ª, "Total" a 5 821ª;
- nome é o contrário: "Juliana" é a 7 294ª, "juliana" a 59 329ª.

A razão entre as duas posições separa os grupos:

| | Razão posição(Maiúscula) / posição(minúscula) |
|---|---|
| Palavras comuns | Total 13 · Ruim 22 · Mãe 8,4 · Ontem 3,9 · Escola 1,9 |
| Nomes, lugares, marcas | Juliana 0,12 · Canoas 0,49 · Instagram 0,32 |
| Ambíguos | Irmã 6,9 · Domingo 6,6 · Vitória 2,3 · Glória 1,7 · Rosa 0,68 |

O filtro aplica, em ordem:
1. termo liberado → descartado;
2. pronome de tratamento ("Dona", "Doutor") sai do trecho. Se o NER engoliu a frase
   ("Saudade da minha"), o trecho é cortado na primeira palavra minúscula que não é
   conectivo;
3. no início de frase ou de linha, onde a maiúscula não informa nada: se todas as
   palavras são comuns pelo léxico → descartado;
4. no meio da frase, uma palavra sozinha fora da lista de nomes e muito comum →
   descartada;
5. o que o NER marcou em minúsculas ("lucas") fica: é sinal forte de nome.

**Na dúvida, censura.** O que sobra de mais aparece numa revisão; o que escapa, não.

### Marcadores e sobreposições

- **Numeração.** O mesmo valor recebe sempre o mesmo marcador, sem diferenciar
  maiúsculas, acentos e espaços. A numeração segue a ordem de aparição.
  `[PACIENTE]` não é numerado.
- **Sobreposições.** Detecções que se sobrepõem viram a união: censurar a mais
  nunca deixa escapar um pedaço. O tipo do trecho fundido é decidido em ordem:
  1. um dado estruturado (e-mail, CPF, telefone, CEP, data) que cobre o trecho
     inteiro vence;
  2. senão, `[PACIENTE]`, se as partes do paciente cobrem mais da metade. Assim,
     "Dona Ana Souza" continua sendo a paciente, mas "Carlos Souza" (mesmo
     sobrenome) é outra pessoa;
  3. senão, o pedaço mais longo. No empate, vale a camada mais confiável: o `md`
     chama "Juliana" de lugar, e o dicionário de nomes sabe que é pessoa.
- **`reverter`** tolera as variações que uma LLM costuma devolver: `PESSOA_1`,
  `[Pessoa_1]`, `[pessoa 1]`.

### Garantias de privacidade

- **O mapa só existe em memória**: a biblioteca não grava, não loga e não envia.
  Os campos com conteúdo do resultado têm `repr=False`, então um `print(resultado)`
  não despeja o texto num log.
- **Logs só com metadados** (tamanho, contagem por tipo, tempo). Um teste roda
  todos os fixtures e procura, no log, na saída e nos avisos do Python, os dados
  sensíveis e as frases deles.
- **Nada de rede:**
  - o spaCy é carregado por nós e injetado. O `load()` do Presidio chamaria
    `spacy.cli.download` se não achasse o modelo;
  - o registro de reconhecedores nunca fica vazio. Vazio, o Presidio carrega os
    reconhecedores em inglês, e os de e-mail e URL usam o `tldextract`, que baixa
    a lista de sufixos. Há teste conferindo o conjunto exato.
- **`log_decision_process=False`.** Ligado, o Presidio registra os trechos
  encontrados num logger próprio.

### Por que não o `presidio-anonymizer`

- Ele prende o `cryptography` abaixo da versão 49.
- O operador customizado dele recebe só o texto, sem tipo nem posição.
- A resolução de conflitos dele refaria a fusão de trechos descrita acima.

## Medições

Sobre os 27 textos fictícios anotados de `tests/fixtures/`. Recall = ocorrências
censuradas / esperadas, pela camada que tem a obrigação de pegar o caso. Detalhes
em [`docs/medicoes.md`](docs/medicoes.md).

| Modelo | Camada 1 | Camada 2 | Camada 3 | Camada 4 (só NER) | ~60 min de fala (8 mil palavras) | Falsos positivos nos fixtures |
|---|---|---|---|---|---|---|
| `pt_core_news_md` | 58/58 | 18/18 | 50/50 | 29/31 | 1,3 s | nenhum |
| `pt_core_news_lg` | 58/58 | 18/18 | 50/50 | 29/31 | 1,6 s | nenhum |

## Limitações

- **Apelidos** ("Juninho", "Tati") e **nomes em minúsculas** numa transcrição sem
  pontuação dependem só do NER, e às vezes escapam.
- **A lista do IBGE tem ~1 000 nomes**, não a base inteira do Censo: a API não
  exporta tudo. Os nomes raros ficam com o NER.
- **O spaCy não dá confiança por entidade.** O limiar do NER, na prática, liga ou
  desliga a camada.
- **Os fixtures são poucos e escritos à mão.** As medições acima não são uma
  avaliação em dados reais.

**Use com revisão humana.** O anonimizador reduz muito o trabalho, mas não substitui
alguém conferindo o texto antes do envio.

## Testes

```bash
pip install -e ".[dev]"
pytest                                                # rápidos: sem modelo, sem rede
ANONIMIZADOR_MODELO=pt_core_news_md pytest -m modelo -s   # com o modelo de verdade
```

**Os testes rápidos:**
- usam um pipeline spaCy em branco (camadas 1 a 3);
- imitam o NER com um `entity_ruler` que rotula exatamente o que o spaCy errava;
- usam um léxico com as posições medidas no modelo.

**Os fixtures** (`tests/fixtures/*.json`) são textos fictícios. Cada um traz:
- os dados do paciente;
- o que precisa sair censurado, com a camada responsável;
- o que precisa continuar no texto.

## Ferramentas

- `ferramentas/medir_censura.py <modelo> [<modelo> ...]` — regera `docs/medicoes.md`.
- `ferramentas/gerar_nomes_ibge.py` — regera `anonimizador/dados/nomes_ibge.txt`
  pela API de Nomes do IBGE (`pip install -e ".[ferramentas]"`). Junta os rankings
  por UF, sexo e década com os candidatos de `nomes_candidatos.txt` que o Censo
  confirma.

## Estrutura

```
anonimizador/
  __init__.py        anonimizar() e o resultado
  motor.py           Presidio em português: registro, spaCy injetado, trava
  reconhecedores.py  as quatro camadas
  marcadores.py      sobreposições, marcadores e reverter (Python puro)
  lexico.py          frequência de palavras a partir dos vetores do spaCy
  texto.py           normalização que preserva as posições
  dados/             nomes do IBGE, nomes ambíguos, termos liberados
tests/               pytest + fixtures fictícios anotados
ferramentas/         medições e geração da lista de nomes
```

## Licença e créditos

Código sob a licença [MIT](LICENSE).

- [Presidio Analyzer](https://github.com/microsoft/presidio) — MIT, Microsoft.
- [spaCy](https://github.com/explosion/spaCy) — MIT, Explosion.
- Modelos `pt_core_news_md`/`lg` — CC BY-SA 4.0. **Não fazem parte deste
  repositório**: são instalados à parte pelo `spacy download`.
- Lista de nomes (`anonimizador/dados/nomes_ibge.txt`) — derivada da
  [API de Nomes do IBGE](https://servicodados.ibge.gov.br/api/docs/nomes?versao=2)
  (Censo Demográfico 2010).
