"""
Frequência de palavras, para separar "palavra comum com maiúscula" de "nome".

O NER do spaCy português marca como pessoa ou lugar quase tudo que começa com
maiúscula — e numa transcrição toda linha começa com maiúscula: "Total.",
"Ansiedade. Medo.", "Saudade da minha mãe" viravam [PESSOA_n].

A pista para separar vem do próprio modelo: o vocabulário dos vetores do spaCy pt
diferencia maiúsculas e está em ordem de frequência (igual no `md` e no `lg`). Uma
palavra comum aparece muito mais em minúsculas ("total" é a 440ª, "Total" a
5 821ª); um nome, ao contrário ("Juliana" é a 7 294ª, "juliana" a 59 329ª). A
razão entre as duas posições separa os grupos com folga — números medidos no
`pt_core_news_md`:

    palavras comuns: Total 13 · Ruim 22 · Mãe 8,4 · Ontem 3,9 · Escola 1,9
    nomes, lugares, marcas: Juliana 0,12 · Canoas 0,49 · Instagram 0,32
    nomes ambíguos: Irmã 6,9 · Domingo 6,6 · Vitória 2,3 · Glória 1,7 · Rosa 0,68

Sem vetores (o `spacy.blank` dos testes) o léxico fica vazio e não opina: nenhuma
palavra é tratada como comum, e nada deixa de ser censurado por causa dele.
"""
import math

import numpy as np


def _capitalizada(palavra: str) -> str:
    return palavra[:1].upper() + palavra[1:].lower()


class Lexico:
    def __init__(self, posicao):
        # `posicao(palavra)` → posição no ranking de frequência, ou None se a
        # palavra não está no vocabulário.
        self._posicao = posicao

    @classmethod
    def vazio(cls) -> "Lexico":
        return cls(lambda _palavra: None)

    @classmethod
    def de_posicoes(cls, posicoes: dict[str, int]) -> "Lexico":
        """Para os testes: as posições escritas à mão."""
        return cls(posicoes.get)

    @classmethod
    def do_spacy(cls, nlp) -> "Lexico":
        """
        Ranking a partir da ordem das chaves dos vetores. Busca binária num array
        ordenado (~4 MB para 500 mil chaves) em vez de um dict, que ocuparia dez
        vezes isso.
        """
        from spacy.strings import get_string_id

        chaves_por_linha = nlp.vocab.vectors.key2row
        if not chaves_por_linha:
            return cls.vazio()
        chaves = np.fromiter(chaves_por_linha.keys(), dtype=np.uint64, count=len(chaves_por_linha))
        ordem = np.argsort(chaves, kind="stable")
        ordenadas = chaves[ordem]

        def posicao(palavra: str) -> int | None:
            chave = np.uint64(get_string_id(palavra))
            indice = int(np.searchsorted(ordenadas, chave))
            if indice < len(ordenadas) and ordenadas[indice] == chave:
                return int(ordem[indice])
            return None

        return cls(posicao)

    def razao(self, palavra: str) -> float | None:
        """
        Posição da forma Capitalizada / posição da minúscula. Quanto maior, mais a
        palavra é usada em minúsculas — mais "comum". None quando a minúscula não
        está no vocabulário (palavra rara, provavelmente nome).
        """
        minuscula = self._posicao(palavra.lower())
        if minuscula is None:
            return None
        capitalizada = self._posicao(_capitalizada(palavra))
        if capitalizada is None:
            return math.inf
        return capitalizada / max(minuscula, 1)

    def eh_palavra_comum(self, palavra: str, limiar: float) -> bool:
        razao = self.razao(palavra)
        return razao is not None and razao >= limiar
