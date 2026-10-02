# Medições

Geradas por `ferramentas/medir_censura.py` — não editar à mão; rodar de novo.

- Data: 2026-10-02
- Máquina: Windows-10-10.0.19045-SP0, 24 threads; Python 3.14.5, spaCy 3.8.16
- Textos: os 27 fixtures fictícios de `tests/fixtures/`

Recall = ocorrências censuradas / ocorrências esperadas, pela camada que tem a
obrigação de pegar o caso. "60 min de fala" = o fixture 22 repetido até ~8 mil palavras.

| Modelo | Camada 1 (deny-list) | Camada 2 (regex) | Camada 3 (IBGE) | Camada 4 (só NER) | 60 min de fala | Carga |
|---|---|---|---|---|---|---|
| pt_core_news_md 3.8.0 | 58/58 | 18/18 | 50/50 | 29/31 | 1.32 s (8080 palavras) | 1.2 s |
| pt_core_news_lg 3.8.0 | 58/58 | 18/18 | 50/50 | 29/31 | 1.56 s (8080 palavras) | 1.4 s |

**pt_core_news_md 3.8.0 — escaparam:** 09_apelidos.json: “Juninho” (camada 4); 10_transcricao_sem_pontuacao.json: “lucas” (camada 4)

**pt_core_news_md 3.8.0 — censurados sem necessidade:** nada

**pt_core_news_lg 3.8.0 — escaparam:** 09_apelidos.json: “Tati” (camada 4); 10_transcricao_sem_pontuacao.json: “lucas” (camada 4)

**pt_core_news_lg 3.8.0 — censurados sem necessidade:** nada
