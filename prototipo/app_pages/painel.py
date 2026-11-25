"""Painel da unidade: lista dos estudantes de uma unidade ordenada pelo escore, com a ficha de cada um.

Pensado para a equipe de assistência estudantil no início de cada ciclo (entrada de uma turma): os dados usados pelo
modelo são do ingresso, então a lista indica, desde o começo, quem merece acompanhamento mais próximo.
"""
import altair as alt
import pandas as pd
import streamlit as st

import comum as C
import nucleo as N

FONTES = {'demo': 'Demonstração', 'csv': 'Arquivo CSV', 'extraido': 'Extraídos'}
PERFIL = ('Idade_Inicio_Estimada', 'Sexo', 'Tipo de Deficiencia')

modelo, esquema, chave_modelo = C.modelo_ou_parar()
lim_m, lim_a = C.limites(esquema)
resultados = N.carregar_resultados() or {}


def _limpar(*chaves):
    for k in chaves:
        st.session_state.pop(k, None)


def _ao_mudar_fonte():
    _limpar('inst', 'uni', 'curso', 'nivel', 'tabela')


# ---------------------------------------------------------------- barra lateral: origem dos dados
if 'fonte_desejada' in st.session_state:          # vindo da página de extração
    _ao_mudar_fonte()
    st.session_state['fonte'] = st.session_state.pop('fonte_desejada')
opcoes = [f for f in FONTES if f != 'extraido' or C.EXTRAIDO[0] in st.session_state]
if st.session_state.get('fonte') not in opcoes:
    st.session_state['fonte'] = 'demo'

with st.sidebar:
    st.subheader(':material/database: Dados')
    fonte = st.segmented_control('Origem dos dados', opcoes, required=True, format_func=FONTES.get, key='fonte',
                                 on_change=_ao_mudar_fonte, width='stretch', persist_state='session')
    arquivo = None
    if fonte == 'demo':
        st.caption('Matrículas reais do conjunto de teste do modelo (15 unidades de 5 institutos federais), sem '
                   'desfecho e sem identificação.')
    elif fonte == 'extraido':
        st.caption(f'{st.session_state[C.EXTRAIDO[1]]}. Para outro recorte, use a página "Extrair dos microdados".')
    else:
        arquivo = st.file_uploader('Lista de estudantes (CSV)', type=['csv'], key='arquivo',
                                   on_change=_ao_mudar_fonte,
                                   help='Uma linha por estudante, com as colunas do modelo de CSV. Separador vírgula '
                                        'ou ponto e vírgula; UTF-8 ou Latin-1.')
        exemplos = pd.read_csv(N.DEMO_FILE, encoding='utf-8-sig').head(3) if N.DEMO_FILE.exists() else None
        st.download_button('Baixar modelo de CSV', N.modelo_csv(esquema, exemplos), 'modelo_entrada_ews.csv',
                            mime='text/csv', icon=':material/download:', width='stretch')
        with st.expander('Colunas esperadas', icon=':material/view_column:'):
            st.dataframe(pd.DataFrame({'Coluna': [N.COLUNA_ID] + [N.nome_csv(c) for c in esquema['colunas']],
                                       'Tipo': ['texto (opcional)'] + ['texto' if c in esquema['categoricas']
                                                                       else 'número' for c in esquema['colunas']]}),
                         hide_index=True, alt='Colunas do CSV e seus tipos')
            st.caption(f'"{N.COLUNA_ID}" (matrícula ou nome) só é exibida, nunca usada pelo modelo. '
                       f'"{N.IDADE_CSV}" é a idade no ingresso. "Duracao_Ciclo_Dias" pode ser substituída por '
                       f'"{N.DATAS_CICLO[0]}" e "{N.DATAS_CICLO[1]}". Campos vazios contam como não informados.')


def _parar_com_erro(mensagem, dica=None):
    st.title('Painel da unidade', anchor=False)
    st.error(mensagem, icon=':material/error:')
    if dica:
        st.caption(dica)
    st.stop()


# ---------------------------------------------------------------- leitura e preparo
if fonte == 'extraido':
    bruto = st.session_state[C.EXTRAIDO[0]]
    X_total, avisos, ids = N.preparar_entradas(bruto, esquema)
    avisos = list(st.session_state.get(C.EXTRAIDO[2], [])) + avisos
    origem = st.session_state[C.EXTRAIDO[1]]
else:
    if fonte == 'csv':
        if arquivo is None:
            st.title('Painel da unidade', anchor=False)
            st.info('Envie a lista de estudantes na barra lateral ou volte aos dados de demonstração. O modelo de CSV '
                    'traz os nomes de coluna esperados e três linhas de exemplo.', icon=':material/upload_file:')
            st.stop()
        conteudo, origem = arquivo.getvalue(), f'arquivo {arquivo.name}'
    else:
        if not N.DEMO_FILE.exists():
            _parar_com_erro(f'Arquivo de demonstração ausente: {N.DEMO_FILE.name}.')
        conteudo, origem = N.DEMO_FILE.read_bytes(), 'demonstração com dados reais do teste do modelo'
    try:
        bruto = N.ler_csv(conteudo)
        X_total, avisos, ids = N.preparar_entradas(bruto, esquema)
    except N.ErroImportacao as erro:
        _parar_com_erro(str(erro), 'Confira o arquivo com o modelo de CSV e o quadro "Colunas esperadas".')
    except Exception as erro:  # noqa: BLE001
        _parar_com_erro(f'Não foi possível ler o arquivo. Detalhe técnico: {erro}')
ciclos = N.detalhes_ciclo(bruto, esquema)   # turma e datas, quando o arquivo as traz (só exibidas na ficha)

def _evasao_por_nivel():
    niveis = resultados.get('niveis_teste')
    if not niveis:
        return ''
    taxas = ', '.join(f'{N.fmt_num(niveis[nv]["evasao_observada_pct"], 0)}%' for nv in reversed(N.NIVEIS))
    return f' Nas turmas de teste, evadiram {taxas} dos estudantes dos níveis baixo, médio e alto.'


# ---------------------------------------------------------------- barra lateral: unidade
with st.sidebar:
    st.subheader(':material/school: Unidade')
    inst = st.selectbox('Instituição', sorted(X_total['Instituição'].unique()), key='inst', bind='query-params',
                        on_change=_limpar, args=('uni', 'curso', 'nivel', 'tabela'))
    unidades = N.filtrar(X_total, inst)['Unidade de Ensino'].value_counts()
    uni = st.selectbox('Unidade de ensino', sorted(unidades.index), key='uni', bind='query-params',
                       format_func=lambda u: f'{u} ({unidades[u]})',
                       on_change=_limpar, args=('curso', 'nivel', 'tabela'))
    st.divider()
    with st.expander('Como ler este painel', icon=':material/help:'):
        st.markdown(
            '- **Quando usar:** no início de cada ciclo, com a lista de ingressantes. O modelo usa só dados do '
            'ingresso, então o escore de um estudante não muda ao longo do curso.\n'
            '- **O que o modelo vê:** idade, sexo e deficiência do estudante, além do curso, da oferta e da unidade. '
            'Notas, frequência e situação de vida não entram; a conversa com o estudante continua essencial.\n'
            '- **Escore:** de 0 a 1. Quanto maior, mais o perfil se parece com o de quem evadiu em turmas '
            'anteriores. Serve para ordenar a lista; não é a probabilidade de evadir.\n'
            f'- **Níveis:** {C.texto_niveis(esquema)}. São faixas experimentais.{_evasao_por_nivel()}\n'
            '- **Precisão:** dentro de uma unidade, a ordenação é modesta (ROC-AUC de cerca de 0,64). Use a lista '
            'como ponto de partida, não como veredito.\n'
            '- **Fatores:** mostram o que o modelo associou à evasão nos dados históricos, não as causas da evasão '
            'deste estudante.\n'
            '- **Uso:** apoio para organizar o acompanhamento. Não substitui a escuta do estudante nem justifica '
            'decisões automáticas.')
    st.caption('Protótipo acadêmico de execução local; nada é armazenado.')

X = N.filtrar(X_total, inst, uni)
escore, shap = C.pontuar(modelo, X, esquema, chave_modelo)
nivel = N.nivel_alerta(escore, lim_m, lim_a)
tabela = N.tabela_estudantes(X, escore, nivel, shap, esquema, ids)
n = len(X)

# ---------------------------------------------------------------- cabeçalho e indicadores
st.title(uni, anchor=False)
st.caption(f'{inst} · {N.fmt_num(n, 0)} estudantes de cursos técnicos · {origem}')
st.caption(':material/info: Lista ordenada pelo escore do modelo, calculado com dados do ingresso. Use-a para '
           'decidir por onde começar o acompanhamento ao longo do ciclo; o nível não é diagnóstico.')
if avisos:
    st.warning('\n'.join(f'- {a}' for a in avisos), title='Atenção aos dados', icon=':material/warning:')

with st.container(horizontal=True):
    st.metric('Estudantes', N.fmt_num(n, 0), f'{X["Nome de Curso"].nunique()} curso(s)', delta_color='off',
              delta_arrow='off', border=True, icon=':material/groups:')
    for nv in N.NIVEIS:
        k = int((nivel == nv).sum())
        st.metric(f'Nível {nv.lower()}', N.fmt_num(k, 0), f'{N.fmt_num(100 * k / n, 0)}% da unidade',
                  delta_color='off', delta_arrow='off', border=True, icon=C.ICONE_NIVEL[nv])


# ---------------------------------------------------------------- ficha do estudante
SECOES = {'estudante': 'Estudante', 'curso': 'Curso, oferta e ciclo', 'instituicao': 'Instituição'}


def perfil_completo(explicacao, ciclo):
    """Todos os atributos do estudante, por grupo e na ordem do esquema, com o efeito de cada um no escore."""
    tabela = explicacao.set_index('coluna').loc[esquema['colunas']]
    if fonte == 'extraido' and st.session_state.get('deficiencia_suposta'):
        tabela.loc['Tipo de Deficiencia', 'Valor'] += ' (suposto)'
    for grupo, titulo in SECOES.items():
        parte = tabela[tabela['Grupo'].eq(grupo)]
        linhas = pd.DataFrame({'Valor': parte['Valor'].to_numpy(),
                               'Efeito': [f'{"+" if v >= 0 else ""}{N.fmt_num(v, 3)}'
                                                    for v in parte['Contribuição']]},
                              index=parte['Atributo'].to_numpy())
        if grupo == 'curso' and not ciclo.empty:   # turma e datas: só exibidas, fora do modelo
            linhas = pd.concat([pd.DataFrame({'Valor': ciclo.to_numpy(), 'Efeito': '—'},
                                             index=ciclo.index), linhas])
        st.markdown(f'**{titulo}**')
        st.table(linhas, border='horizontal', alt=f'Todos os dados do estudante: {titulo.lower()}')
    st.caption('Efeito: contribuição de cada atributo (SHAP, em log-odds); positivo eleva, negativo reduz. '
               'Turma e datas, quando o arquivo as traz, só são exibidas: o modelo usa a duração prevista do ciclo.')


def ficha(numero):
    pos = numero - 1
    linha = X.loc[[pos]]
    i = X.index.get_loc(pos)
    explicacao = N.explicar_registro(linha, shap[i, :-1], esquema)
    dados = tabela[tabela['Nº'].eq(numero)].iloc[0]
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment='center'):
            st.subheader(dados['Estudante'], anchor=False)
            st.badge(f'Nível {dados["Nível"].lower()}', icon=C.ICONE_NIVEL[dados['Nível']],
                     color=C.BADGE_NIVEL[dados['Nível']])
        st.caption(f'{linha["Nome de Curso"].iat[0]} · {linha["Tipo de Oferta"].iat[0]} · '
                   f'{N.fmt_valor("Turno", linha["Turno"].iat[0])}  \n'
                   f'{dados["Posição"]}º de {n} na unidade · escore {N.fmt_num(dados["Escore"], 2)}')
        st.markdown('**O que mais pesou no escore**')
        st.markdown(N.resumo_textual(explicacao))
        top = explicacao.head(6).assign(Rótulo=lambda t: t['Atributo'] + ': ' + t['Valor'])
        st.altair_chart(alt.Chart(top).mark_bar(cornerRadius=3).encode(
            x=alt.X('Contribuição:Q', title='Contribuição'),
            # Ordem explícita: sem ela, ao trocar de estudante o gráfico mantinha a ordem das barras anteriores.
            y=alt.Y('Rótulo:N', sort=top['Rótulo'].tolist(), title=None, axis=alt.Axis(labelLimit=230, labelOverlap=False)),
            color=alt.Color('Efeito:N', scale=alt.Scale(domain=list(C.COR_EFEITO), range=list(C.COR_EFEITO.values())),
                            legend=alt.Legend(orient='bottom', title=None)),
            tooltip=['Atributo', 'Valor', alt.Tooltip('Contribuição:Q', format='.3f'), 'Efeito'],
        ).properties(height=alt.Step(26)), alt='Atributos que mais pesaram no escore')
        st.markdown('**Dados do estudante usados pelo modelo**')
        perfil = {N.rotulo(c): N.fmt_valor(c, linha[c].iat[0]) for c in PERFIL}
        if fonte == 'extraido' and st.session_state.get('deficiencia_suposta'):
            perfil[N.rotulo('Tipo de Deficiencia')] += ' (suposto)'   # ausente na edição 2024 dos microdados
        st.table(perfil, border='horizontal', alt='Dados do estudante usados pelo modelo')
        with st.expander('Perfil completo e dados do ciclo', icon=':material/id_card:'):
            perfil_completo(explicacao, ciclos.loc[pos])
        st.caption('Os fatores mostram como o modelo chegou ao escore, a partir de padrões de turmas anteriores. '
                   'Não são causas da evasão nem um diagnóstico do estudante.')


# ---------------------------------------------------------------- abas
aba_lista, aba_cursos = st.tabs([':material/table_rows: Estudantes', ':material/bar_chart: Visão por curso'])

with aba_lista:
    col_lista, col_ficha = st.columns([5, 3], gap='medium')
    with col_lista, st.container(border=True):
        cursos = sorted(X['Nome de Curso'].unique())
        with st.container(horizontal=True, vertical_alignment='bottom'):
            curso = st.selectbox('Curso', [N.TODOS] + cursos if len(cursos) > 1 else cursos, key='curso',
                                 on_change=_limpar, args=('tabela',),
                                 format_func=lambda c: c.replace('Técnico em ', ''))
            base_curso = tabela if curso == N.TODOS else tabela[tabela['Curso'].eq(curso.replace('Técnico em ', ''))]
            filtro = st.segmented_control(
                'Nível', ['Todos', *N.NIVEIS], default='Todos', required=True, key='nivel',
                format_func=lambda v: f'{v} ({len(base_curso) if v == "Todos" else int(base_curso["Nível"].eq(v).sum())})',
                on_change=_limpar, args=('tabela',))
        visivel = (base_curso if filtro == 'Todos' else base_curso[base_curso['Nível'].eq(filtro)]).reset_index(drop=True)
        colunas = ['Posição', 'Estudante', 'Nível', 'Escore', 'Curso', 'Fator principal']
        if len(cursos) == 1 or curso != N.TODOS:
            colunas.remove('Curso')
        evento = st.dataframe(
            C.colorir_nivel(visivel), hide_index=True, on_select='rerun', selection_mode='single-row', key='tabela',
            height=520, column_order=colunas, alt='Estudantes da unidade ordenados pelo escore',
            column_config={
                'Posição': st.column_config.NumberColumn('#', width=35, help='Ordem na unidade pelo escore.'),
                'Estudante': st.column_config.TextColumn(width=90),
                'Nível': st.column_config.TextColumn(width=55),
                'Escore': st.column_config.ProgressColumn(width=75, min_value=0.0, max_value=1.0, format='%.2f',
                                                          help='De 0 a 1; ordena a lista. Não é probabilidade.'),
                'Curso': st.column_config.TextColumn(width=105),
                'Fator principal': st.column_config.TextColumn(
                    width=260, help='Atributo que mais elevou o escore deste estudante.'),
            })
        with st.container(horizontal=True, vertical_alignment='center'):
            st.caption('Clique em uma linha para abrir a ficha. Sem seleção, mostra-se o primeiro da lista.')
            st.download_button('Baixar lista', visivel[['Posição', 'Estudante', 'Nível', 'Escore', 'Curso', 'Oferta',
                                                        'Turno', 'Fator principal']]
                               .to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig'),
                               f'lista_{inst}_{uni}.csv'.replace(' ', '_'), mime='text/csv',
                               icon=':material/download:', type='tertiary')
    linhas = evento.selection.rows if evento is not None else []
    numero = N.registro_selecionado(visivel, linhas)
    with col_ficha:
        if numero is None:
            st.info('Nenhum estudante neste filtro.', icon=':material/filter_alt_off:')
        else:
            ficha(numero)

with aba_cursos:
    with st.container(border=True):
        st.subheader('Níveis por curso', anchor=False)
        por_curso = (tabela.groupby(['Curso', 'Nível']).size().rename('Estudantes').reset_index())
        ordem = (tabela.assign(alto=tabela['Nível'].eq('Alto')).groupby('Curso')['alto'].mean()
                 .sort_values(ascending=False, kind='stable').index.tolist())
        st.altair_chart(alt.Chart(por_curso).mark_bar().encode(
            x=alt.X('Estudantes:Q', title='Estudantes'),
            y=alt.Y('Curso:N', sort=ordem, title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
            color=alt.Color('Nível:N', scale=alt.Scale(domain=list(N.NIVEIS), range=[C.COR_NIVEL[v] for v in N.NIVEIS]),
                            legend=alt.Legend(orient='bottom', title=None)),
            order=alt.Order('ordem:Q'),
            tooltip=['Curso', 'Nível', 'Estudantes'],
        ).transform_calculate(ordem="datum['Nível'] == 'Alto' ? 0 : datum['Nível'] == 'Médio' ? 1 : 2")
            .properties(height=alt.Step(30)), alt='Estudantes por nível de alerta em cada curso')
        st.caption('Cursos ordenados pela proporção de estudantes no nível alto. Ajuda a planejar ações coletivas '
                   '(por exemplo, numa turma inteira), além do acompanhamento individual.')
    with st.container(border=True):
        st.subheader('Fatores mais frequentes no nível alto', anchor=False)
        altos = tabela[tabela['Nível'].eq('Alto')]
        if altos.empty:
            st.caption('Nenhum estudante no nível alto nesta unidade.')
        else:
            fatores = (altos['Fator principal'].str.split(':').str[0].value_counts().head(8)
                       .rename_axis('Fator').reset_index(name='Estudantes'))
            st.altair_chart(alt.Chart(fatores).mark_bar(cornerRadius=3, color=C.COR_NIVEL['Alto']).encode(
                x=alt.X('Estudantes:Q', title='Estudantes do nível alto'),
                y=alt.Y('Fator:N', sort=fatores['Fator'].tolist(), title=None, axis=alt.Axis(labelLimit=300, labelOverlap=False)),
                tooltip=['Fator', 'Estudantes'],
            ).properties(height=alt.Step(26)),
                alt='Atributos que mais elevaram o escore dos estudantes de nível alto')
            st.caption('Atributo que mais elevou o escore de cada estudante de nível alto. Quando predominam '
                       'atributos do curso ou da oferta (tipo de oferta, turno), o alerta é sobretudo da turma, e ações '
                       'coletivas tendem a fazer mais sentido.')
