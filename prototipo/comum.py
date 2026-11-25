"""Elementos de interface compartilhados pelas páginas: modelo em cache, cálculo dos escores e cores."""
import numpy as np
import pandas as pd
import streamlit as st

import nucleo as N

COR_NIVEL = {'Alto': '#c2410c', 'Médio': '#d99a1e', 'Baixo': '#5b8fd1'}
FUNDO_NIVEL = {'Alto': '#fde3d3', 'Médio': '#fdf1cf', 'Baixo': '#e1ecfa'}
BADGE_NIVEL = {'Alto': 'red', 'Médio': 'orange', 'Baixo': 'blue'}
ICONE_NIVEL = {'Alto': ':material/priority_high:', 'Médio': ':material/schedule:', 'Baixo': ':material/check_circle:'}
COR_EFEITO = {'Eleva o escore': '#eb6834', 'Reduz o escore': '#2a78d6'}
COR_GRUPO = {'Estudante': '#2a78d6', 'Curso e oferta': '#8a8a85', 'Instituição': '#d99a1e'}

# Chaves de st.session_state usadas para levar um recorte da página de extração ao painel.
EXTRAIDO = ('dados_extraidos', 'descricao_extraida', 'notas_extraidas')


@st.cache_resource(show_spinner=False)
def _modelo_e_esquema():
    return N.carregar_modelo_e_esquema()


def modelo_ou_parar():
    """(modelo, esquema, chave) ou, se os arquivos faltarem ou não combinarem, uma mensagem e st.stop()."""
    if not N.MODELO_FILE.exists() or not N.ESQUEMA_FILE.exists():
        st.error(f'Faltam {N.MODELO_FILE.name} e/ou {N.ESQUEMA_FILE.name} em resultados/prototipo/ (gerados pelo '
                 'notebook, seção 10).',
                 icon=':material/error:')
        st.stop()
    try:
        modelo, esquema = _modelo_e_esquema()
    except Exception as erro:  # noqa: BLE001
        st.error(f'Falha ao carregar o modelo ou o esquema: {erro}', icon=':material/error:')
        st.stop()
    estado = N.MODELO_FILE.stat()
    return modelo, esquema, f'{estado.st_mtime_ns}-{estado.st_size}'


@st.cache_data(show_spinner=False, max_entries=32)
def pontuar(_modelo, X: pd.DataFrame, _esquema: dict, chave_modelo: str):
    """Escores e matriz SHAP (última coluna = valor-base) de um conjunto de matrículas."""
    if X.empty:
        return np.array([]), np.zeros((0, len(_esquema['colunas']) + 1))
    return _modelo.predict_proba(X)[:, 1], N.shap_valores(_modelo, X, _esquema)


def limites(esquema):
    return esquema['limiares']['medio'], esquema['limiares']['alto']


def texto_niveis(esquema):
    lim_m, lim_a = limites(esquema)
    return (f'Baixo (escore abaixo de {N.fmt_num(lim_m, 2)}), Médio ({N.fmt_num(lim_m, 2)} a '
            f'{N.fmt_num(lim_a, 2)}) e Alto (a partir de {N.fmt_num(lim_a, 2)})')


def colorir_nivel(tabela: pd.DataFrame, coluna='Nível'):
    """Styler que pinta o fundo da coluna de nível."""
    return tabela.style.map(lambda v: f'background-color: {FUNDO_NIVEL.get(v, "")}', subset=[coluna])
