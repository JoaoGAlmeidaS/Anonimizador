"""
Normalização de texto para comparar sem diferenciar maiúsculas e acentos.

O ponto delicado é preservar as posições: o casamento é feito numa cópia
normalizada, mas o trecho censurado é recortado do texto original pelas mesmas
posições. Por isso a troca é feita caractere a caractere, sempre um por um —
`unicodedata.normalize("NFD", texto)` no texto inteiro desalinharia tudo, porque
"ç" vira dois caracteres.
"""
import unicodedata
from functools import lru_cache


@lru_cache(maxsize=4096)
def _normalizar_caractere(caractere: str) -> str:
    base = unicodedata.normalize("NFD", caractere)[0]
    minuscula = base.lower()
    # Raros, mas existem: caracteres cuja minúscula tem dois pontos de código
    # ("İ" → "i̇"). Ficam como estão para não deslocar as posições.
    return minuscula if len(minuscula) == 1 else base


def normalizar(texto: str) -> str:
    """Minúsculo e sem acento, com exatamente o mesmo comprimento do original."""
    return "".join(_normalizar_caractere(caractere) for caractere in texto)


def chave(valor: str) -> str:
    """Identidade de um valor censurado: "Carlos" e "carlos" são a mesma pessoa."""
    return " ".join(normalizar(valor).split())
