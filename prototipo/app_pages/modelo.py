"""Como o modelo decide: dados de entrada, desempenho, principal limitação e decomposição de um escore.

Página técnica, para quem quer entender o modelo além do painel. Os números da avaliação vêm de resultados_modelo.json,
gravado pelo notebook; a importância dos atributos e a decomposição são calculadas aqui, na base de demonstração.
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import comum as C
import nucleo as N

modelo, esquema, chave_modelo = C.modelo_ou_parar()
R = N.carregar_resultados()
pct = lambda v, casas=0: f'{N.fmt_num(v, casas)}%'   # noqa: E731

st.title('Como o modelo decide', anchor=False)
st.caption('Resumo técnico do modelo usado no painel: o que ele recebe, quanto acerta, onde falha e como cada escore '
           'é explicado.')
if R is None:
    st.warning(f'Arquivo {N.RESULTADOS_FILE.name} ausente: os números da avaliação não podem ser exibidos. Ele é '
               'gerado pelo notebook (seção 10).', icon=':material/warning:')
    st.stop()

# ---------------------------------------------------------------- 1. entrada
with st.container(border=True):
    st.subheader('1. O que o modelo recebe', anchor=False)
    pop, part = R['populacao'], R['particoes']
    st.markdown(
        f'Um classificador **CatBoost** (árvores de decisão em sequência) treinado com '
        f'{N.fmt_num(pop["matriculas"], 0)} matrículas de cursos técnicos da Rede Federal que já concluíram ou '
        f'evadiram ({pct(pop["evasao_pct"], 1)} evadiram), dos Microdados de Eficiência Acadêmica da PNP, ano-base '
        f'2025. As turmas foram divididas em treino ({N.fmt_num(part["Treino"]["turmas"], 0)}), validação '
        f'({N.fmt_num(part["Validação"]["turmas"], 0)}) e teste ({N.fmt_num(part["Teste"]["turmas"], 0)}): uma turma '
        'inteira fica num só lado, para que o teste simule turmas novas. O modelo usa '
        f'**{len(esquema["colunas"])} atributos conhecidos no ingresso**:')
    grupos = {'Estudante': esquema['grupos']['estudante'], 'Instituição': esquema['grupos']['instituicao']}
    grupos['Curso e oferta'] = [c for c in esquema['colunas'] if c not in grupos['Estudante'] + grupos['Instituição']]
    st.table({g: '; '.join(N.rotulo(c) for c in cols) for g, cols in grupos.items()}, border='horizontal',
             alt='Atributos do modelo por grupo')
    st.caption('Ficam de fora o desfecho e suas datas, identificadores, colunas que repetem a unidade (município, '
               'região) e colunas quase sempre vazias. A idade no ingresso é estimada (a coluna original é medida no '
               'fim do ano-base), com incerteza de até cerca de dois anos.')

# ---------------------------------------------------------------- 2. desempenho
with st.container(border=True):
    st.subheader('2. Quanto ele acerta nas turmas de teste', anchor=False)
    comp = pd.DataFrame(R['comparacao_modelos']).T
    comp.index = [i.split('. ', 1)[1] for i in comp.index]
    st.dataframe(comp[['ROC-AUC', 'ROC-AUC dentro da unidade']].map(lambda v: N.fmt_num(v, 2)).rename_axis('Alternativa'),
                 alt='Comparação do modelo com alternativas mais simples')
    st.caption('**ROC-AUC** é a chance de um evadido receber escore maior que um concluinte (0,5 = acaso; 1 = '
               'perfeito). **Dentro da unidade** compara só estudantes da mesma unidade (média de '
               f'{R["uso_na_unidade"]["Todas as matrículas de teste"]["unidades avaliadas"]} unidades), que é o uso '
               'do painel. A regra simples usa a taxa histórica de evasão do tipo de oferta; a regressão logística, '
               'os mesmos atributos do CatBoost.')
    niveis = pd.DataFrame(R['niveis_teste']).T
    with st.container(horizontal=True):
        for nv in reversed(N.NIVEIS):
            st.metric(f'Nível {nv.lower()}', pct(niveis.loc[nv, 'evasao_observada_pct']),
                      f'{N.fmt_num(niveis.loc[nv, "matrículas"], 0)} matrículas', delta_color='off',
                      delta_arrow='off', border=True, icon=C.ICONE_NIVEL[nv],
                      help='Proporção de evadidos entre as matrículas de teste deste nível.')
    st.caption(f'Evasão observada em cada nível no teste. Os níveis são faixas fixas do escore '
               f'({C.texto_niveis(esquema)}), escolhidas para leitura e não otimizadas.')

# ---------------------------------------------------------------- 3. limitação
with st.container(border=True):
    st.subheader('3. Por que cinco campos do cadastro ficam de fora', anchor=False)
    campos = pd.DataFrame(R['campos_nao_informados_pct']).T
    uso = R['uso_na_unidade']
    st.markdown(
        'Forma de ingresso, identidade de gênero, tipo de NEE, renda familiar e cor/raça existem nos microdados, mas '
        'ficaram **vazios ou "Não declarada" muito mais vezes entre os evadidos**, mesmo entre colegas da mesma turma. '
        'A explicação provável é que o cadastro foi completado durante o curso, só para quem continuou. Com esses '
        'campos, o modelo aprenderia que "campo vazio = evasão", um sinal que não existe para um ingressante.')
    col_a, col_b = st.columns([3, 2], gap='medium')
    with col_a:
        campos = campos.rename(index={'Identidade de Gênero': 'Identidade de gênero', 'Renda Familiar': 'Renda familiar',
                                      'Cor / Raça': 'Cor/raça'})
        campos.columns = ['Concluintes (%)', 'Evadidos (%)', 'Diferença na mesma turma (p.p.)']
        st.dataframe(campos.map(lambda v: N.fmt_num(v, 1)).rename_axis('Campo vazio ou "Não declarada"'),
                     alt='Campos vazios por desfecho')
    with col_b:
        completo = uso['Só cadastro completo']
        st.metric('Dentro da unidade, só cadastro completo', N.fmt_num(completo['ROC-AUC dentro da unidade'], 2),
                  f'{N.fmt_num(completo["matrículas"], 0)} matrículas', delta_color='off', delta_arrow='off',
                  border=True, help='ROC-AUC do modelo final nas matrículas com os cinco campos preenchidos.')
    st.caption('A última coluna compara colegas da mesma turma. Num teste exploratório com os cinco '
               'campos, o ROC-AUC subia muito, mas o ganho sumia nas matrículas de cadastro completo (0,70 com os '
               'campos e 0,69 sem eles). O modelo final, sem os campos, também ordena bem esses cadastros '
               f'(ROC-AUC {N.fmt_num(completo["ROC-AUC dentro da unidade"], 2)} dentro da unidade). Os campos podem '
               'voltar quando houver históricos preenchidos já no ingresso.')


# ---------------------------------------------------------------- 4. importância
@st.cache_data(show_spinner='Calculando a importância dos atributos…')
def _base_demo(chave):
    X, _, _ = N.preparar_entradas(N.ler_csv(N.DEMO_FILE.read_bytes()), esquema)
    escore, shap = C.pontuar(modelo, X, esquema, chave)
    return X, escore, shap


with st.container(border=True):
    st.subheader('4. O que mais pesa nos escores', anchor=False)
    X_demo, escore_demo, shap_demo = _base_demo(chave_modelo)
    imp = N.importancia_por_atributo(shap_demo, esquema)
    top = imp.head(12)
    st.altair_chart(alt.Chart(top).mark_bar(cornerRadius=3).encode(
        x=alt.X('Importância:Q', title='Média do tamanho da contribuição (|SHAP|)'),
        y=alt.Y('Atributo:N', sort=top['Atributo'].tolist(), title=None, axis=alt.Axis(labelLimit=220, labelOverlap=False)),
        color=alt.Color('Grupo:N', scale=alt.Scale(domain=list(C.COR_GRUPO), range=list(C.COR_GRUPO.values())),
                        legend=alt.Legend(orient='bottom', title=None)),
        tooltip=['Atributo', 'Grupo', alt.Tooltip('Importância:Q', format='.3f')],
    ).properties(height=alt.Step(22)),
        alt='Doze atributos com maior contribuição média')
    grupos_pct = R['shap_participacao_grupos_pct']
    st.caption(f'Calculado nas {N.fmt_num(len(X_demo), 0)} matrículas da base de demonstração. No teste completo, os '
               f'atributos do estudante respondem por {pct(grupos_pct["Estudante"])} da importância, os da instituição '
               f'por {pct(grupos_pct["Instituição"])} e os do curso por {pct(grupos_pct["Curso e oferta"])}.')

# ---------------------------------------------------------------- 5. decomposição
with st.container(border=True):
    st.subheader('5. Como um escore é decomposto', anchor=False)
    st.caption('Os valores SHAP repartem a saída do modelo entre os atributos. O valor-base (média do modelo) somado '
               'às contribuições dá a saída em log-odds; a função sigmoide a converte no escore de 0 a 1.')
    numero = st.number_input('Registro da base de demonstração', 1, len(X_demo), int(np.argmax(escore_demo)) + 1,
                             key='md_registro', help='Por padrão, o registro de maior escore.')
    i = numero - 1
    base, contrib = float(shap_demo[i, -1]), shap_demo[i, :-1]
    bruto = base + float(contrib.sum())
    sigmoide = 1 / (1 + np.exp(-bruto))
    explicacao = N.explicar_registro(X_demo.iloc[[i]], contrib, esquema)
    col_a, col_b = st.columns([2, 3], gap='medium')
    with col_a:
        st.table({'Valor-base': N.fmt_num(base, 3),
                  f'Soma das {len(contrib)} contribuições': f'{"+" if contrib.sum() >= 0 else ""}'
                                                            f'{N.fmt_num(float(contrib.sum()), 3)}',
                  'Saída (log-odds)': N.fmt_num(bruto, 3),
                  'Sigmoide da saída': N.fmt_num(sigmoide, 4),
                  'Escore calculado pelo modelo': N.fmt_num(float(escore_demo[i]), 4),
                  'Confere?': 'sim' if np.isclose(sigmoide, escore_demo[i], atol=1e-9) else 'NÃO'},
                 border='horizontal', alt='Decomposição do escore do registro escolhido')
    with col_b:
        st.dataframe(explicacao[['Atributo', 'Valor', 'Contribuição']], hide_index=True, height=300,
                     alt='Contribuição de cada atributo para o registro escolhido',
                     column_config={'Contribuição': st.column_config.NumberColumn(format='%+.3f')})

# ---------------------------------------------------------------- 6. limites
with st.container(border=True):
    st.subheader('6. Limites', anchor=False)
    st.markdown(
        '- **Retrospectivo:** avaliado com turmas que já terminaram (ciclos encerrados em 2024), não com estudantes '
        'em curso.\n'
        '- **Desempenho dentro da unidade modesto:** ROC-AUC de cerca de 0,64. No ingresso, colegas de unidade '
        'diferem em pouco além de idade, sexo e deficiência; notas e frequência não estão nos dados públicos.\n'
        '- **Escore e níveis:** o escore ordena, mas não é probabilidade calibrada; os limiares são experimentais.\n'
        '- **Explicações:** SHAP descreve o modelo, não causas da evasão.\n'
        '- **Equidade:** renda e cor/raça não entram no modelo, mas foram usadas para avaliar o desempenho por grupo. '
        'Os grupos "Não declarada" recebem mais alertas falsos; uma avaliação formal de equidade é necessária antes '
        'de qualquer uso institucional.\n'
        '- **Uso:** apoio à equipe, nunca decisão automática.')
