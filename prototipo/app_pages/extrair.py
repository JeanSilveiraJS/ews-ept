"""Extrair dos microdados: recorte real da PNP (edição, instituição, unidade, curso e turmas).

Página técnica: mostra de onde vêm os dados, se cada turma foi usada no treino e, como as turmas já têm
desfecho, compara os níveis do modelo com o que aconteceu. O recorte pode ser aberto no painel ou baixado em CSV.
"""
import re
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st

import comum as C
import extracao as EX
import nucleo as N

modelo, esquema, chave_modelo = C.modelo_ou_parar()
lim_m, lim_a = C.limites(esquema)
SITUACOES = ['Concluintes', 'Evadidos', 'Em curso']


@st.cache_resource(show_spinner=False, max_entries=2)
def _microdados(ano):
    return EX.carregar_tecnicos(ano, esquema)


@st.cache_resource(show_spinner=False)
def _mapa_particao():
    return EX.carregar_mapa_particao()


def _limpar(*chaves):
    for k in ('ex_ciclos', *chaves):
        st.session_state.pop(k, None)


def _abrir_no_painel(dados, descricao, notas, deficiencia_suposta):
    st.session_state['deficiencia_suposta'] = deficiencia_suposta   # edição 2024: tipo de deficiência suposto
    st.session_state[C.EXTRAIDO[0]] = dados
    st.session_state[C.EXTRAIDO[1]] = descricao
    st.session_state[C.EXTRAIDO[2]] = notas
    st.session_state['fonte_desejada'] = 'extraido'


def nome_arquivo(*partes):
    """Nome de arquivo sem acentos nem espaços."""
    texto = unicodedata.normalize('NFKD', '_'.join(str(p) for p in partes)).encode('ascii', 'ignore').decode()
    return re.sub(r'[^A-Za-z0-9_]+', '-', texto).strip('-').lower()[:120] + '.csv'


st.title('Extrair dos microdados', anchor=False)
st.caption('Monte um recorte real da Plataforma Nilo Peçanha (instituição, unidade, curso e turmas) e veja como o '
           'modelo o classifica. Como as turmas já terminaram, dá para comparar os níveis com o que aconteceu. Nada '
           'é gravado; os valores só passam pelas derivações do treinamento (duração do ciclo e idade no ingresso).')

ano = st.segmented_control('Edição dos microdados', list(EX.EDICOES), default=2025, required=True,
                           format_func=lambda a: f'Ano-base {a}', key='ex_ano', bind='query-params',
                           on_change=_limpar, args=('ex_inst', 'ex_uni', 'ex_curso'))
st.caption(EX.EDICOES[ano][2])
try:
    with st.spinner(f'Lendo os microdados do ano-base {ano} (alguns segundos na primeira vez)…'):
        microdados, ausentes, sha_ok = _microdados(ano)
except FileNotFoundError as erro:
    st.error(f'Arquivo de microdados não encontrado: {erro}. Coloque-o na pasta dados/ do projeto (origem e SHA-256 '
             'em dados/LEIA-ME.md).', icon=':material/error:')
    st.stop()
if not sha_ok:
    st.warning('O SHA-256 do arquivo difere do documentado em dados/LEIA-ME.md; os dados podem não ser os do '
               'trabalho.', icon=':material/warning:')

with st.container(horizontal=True):
    inst = st.selectbox('Instituição', sorted(microdados['Instituição'].dropna().unique()), index=None,
                        placeholder='Escolha a instituição', key='ex_inst', bind='query-params',
                        on_change=_limpar, args=('ex_uni', 'ex_curso'))
    base_inst = microdados[microdados['Instituição'].eq(inst)] if inst else microdados.iloc[0:0]
    uni = st.selectbox('Unidade de ensino', sorted(base_inst['Unidade de Ensino'].dropna().unique()), index=None,
                       placeholder='Escolha a unidade', key='ex_uni', bind='query-params', disabled=not inst,
                       on_change=_limpar, args=('ex_curso',))
    base_uni = base_inst[base_inst['Unidade de Ensino'].eq(uni)] if uni else base_inst.iloc[0:0]
    curso = st.selectbox('Curso', [N.TODOS] + sorted(base_uni['Nome de Curso'].dropna().unique()),
                         key='ex_curso', bind='query-params',
                         disabled=not uni, on_change=_limpar)
if not (inst and uni):
    st.info('Escolha a instituição e a unidade para listar as turmas.', icon=':material/filter_alt:')
    st.stop()
recorte = base_uni if curso == N.TODOS else base_uni[base_uni['Nome de Curso'].eq(curso)]
situacoes = st.pills('Situação das matrículas no fim do ano-base', SITUACOES, selection_mode='multi',
                     default=SITUACOES, key='ex_situacoes', on_change=_limpar,
                     help='Concluintes e evadidos têm desfecho conhecido. "Em curso" são matrículas que passaram do '
                          'prazo previsto sem desfecho.')
recorte = recorte[recorte[EX.SITUACAO].isin(situacoes or [])]
if recorte.empty:
    st.info('Nenhuma matrícula com essas situações no recorte.', icon=':material/filter_alt_off:')
    st.stop()

ciclos = EX.resumo_ciclos(recorte, ano, _mapa_particao())
st.subheader(f'Turmas do recorte ({len(ciclos)})', anchor=False)
evento = st.dataframe(
    ciclos, hide_index=True, on_select='rerun', selection_mode='multi-row', key='ex_ciclos',
    alt='Turmas do recorte com datas, oferta, matrículas por situação e uso no modelo',
    column_config={'Ciclo': st.column_config.TextColumn(help='Código do ciclo de matrícula (turma) nos microdados.'),
                   'Início': st.column_config.DateColumn(format='DD/MM/YYYY'),
                   'Fim previsto': st.column_config.DateColumn(format='DD/MM/YYYY'),
                   'Uso no modelo': st.column_config.TextColumn(
                       help='Partição em que a turma ficou no treinamento (só existe para a edição 2025).')})
selecionados = ciclos.iloc[evento.selection.rows]['Ciclo'].tolist() if evento is not None else []
if not selecionados:
    st.info('Marque uma ou mais turmas na tabela para montar o recorte.', icon=':material/touch_app:')
    st.stop()

registros = recorte[recorte[EX.CICLO].isin(selecionados)]
dados, notas = EX.para_formato_prototipo(registros, ano, esquema)
X, avisos, _ = N.preparar_entradas(dados, esquema)
escore, _shap = C.pontuar(modelo, X, esquema, chave_modelo)
nivel = N.nivel_alerta(escore, lim_m, lim_a)
situacao = registros[EX.SITUACAO].to_numpy()
usos = sorted(set(ciclos.loc[ciclos['Ciclo'].isin(selecionados), 'Uso no modelo']))

st.subheader('Recorte escolhido', anchor=False)
with st.container(horizontal=True):
    st.metric('Matrículas', N.fmt_num(len(X), 0), border=True, icon=':material/groups:')
    st.metric('Turmas', N.fmt_num(len(selecionados), 0), border=True, icon=':material/event:')
    for nv in N.NIVEIS:
        st.metric(f'Nível {nv.lower()}', N.fmt_num(int((nivel == nv).sum()), 0), border=True, icon=C.ICONE_NIVEL[nv])
if any(u.startswith(('Treino', 'Validação')) for u in usos):
    st.warning('Há turmas usadas no treinamento do modelo: os escores delas tendem a ser otimistas. Para mostrar '
               'desempenho, prefira turmas do teste.', icon=':material/warning:')
if len(X) < 10:
    st.warning('Recorte com menos de 10 matrículas: há risco de reidentificação, e qualquer comparação fica muito '
               'instável.', icon=':material/visibility:')
for nota in notas + avisos:
    st.caption(f':material/info: {nota}')

with st.container(border=True):
    st.subheader('Níveis do modelo × o que aconteceu', anchor=False)
    cruzada = (pd.crosstab(pd.Categorical(nivel, list(N.NIVEIS)), pd.Categorical(situacao, SITUACOES), dropna=False)
               .rename_axis(index='Nível', columns=None))
    cruzada.index, cruzada.columns = cruzada.index.astype(str), cruzada.columns.astype(str)
    com_desfecho = cruzada['Concluintes'] + cruzada['Evadidos']
    cruzada['Evadiram (entre os com desfecho)'] = (cruzada['Evadidos'] / com_desfecho.where(com_desfecho > 0))
    st.dataframe(cruzada, alt='Matrículas por nível de alerta e situação observada',
                 column_config={'Evadiram (entre os com desfecho)': st.column_config.NumberColumn(format='percent')})
    tem = pd.Series(situacao).isin(['Concluintes', 'Evadidos']).to_numpy()
    evadiu = situacao[tem] == 'Evadidos'
    turmas = registros[EX.CICLO].to_numpy()[tem]
    por_turma = [(int(m.sum()), N.auc(evadiu[m], escore[tem][m])) for m in (turmas == t for t in set(turmas))]
    por_turma = [(k, a) for k, a in por_turma if a is not None]
    auc_todos = N.auc(evadiu, escore[tem])
    auc_turma = (np.average([a for _, a in por_turma], weights=[k for k, _ in por_turma]) if por_turma else None)
    with st.container(horizontal=True):
        st.metric('ROC-AUC no recorte', '—' if auc_todos is None else N.fmt_num(auc_todos, 2), border=True,
                  help='Chance de um evadido do recorte ter escore maior que um concluinte (0,5 = acaso). Exige '
                       'concluintes e evadidos.')
        st.metric('ROC-AUC dentro das turmas', '—' if auc_turma is None else N.fmt_num(auc_turma, 2), border=True,
                  help='O mesmo cálculo feito em cada turma separadamente (média ponderada pelo tamanho): mostra se o '
                       'modelo ordena colegas de uma mesma turma.')
    st.caption('Ilustração, não avaliação: poucas turmas oscilam muito. A avaliação formal é a das 618 turmas de '
               'teste do ano-base 2025 (página "Como o modelo decide"). Matrículas em curso não entram nas contas, '
               'porque o desfecho delas é desconhecido.')

cursos_escolhidos = sorted(registros['Nome de Curso'].dropna().unique())
rotulo_curso = cursos_escolhidos[0] if len(cursos_escolhidos) == 1 else f'{len(cursos_escolhidos)} cursos'
descricao = f'extração: ano-base {ano}, {inst} – {uni}, {rotulo_curso}, {len(selecionados)} turma(s)'
with st.container(horizontal=True, vertical_alignment='center'):
    if st.button('Abrir no painel da unidade', type='primary', icon=':material/groups:',
                 on_click=_abrir_no_painel, args=(dados, descricao, notas, EX.DEFICIENCIA not in registros.columns),
                 key='ex_abrir'):
        # query_params={}: sem ele, switch_page tenta limpar a URL e falha com os parâmetros ligados (ex_inst, ex_uni).
        st.switch_page('app_pages/painel.py', query_params={})
    incluir = st.toggle('Incluir a situação observada no CSV', key='ex_incluir_situacao',
                        help='Acrescenta a coluna "Categoria da Situação", que o painel ignora.')
    st.download_button('Baixar CSV', EX.para_formato_prototipo(registros, ano, esquema, incluir)[0]
                       .to_csv(index=False).encode('utf-8-sig'),
                       nome_arquivo('ews', ano, inst, uni, rotulo_curso, f'{len(selecionados)}turmas'),
                       mime='text/csv', icon=':material/download:')
