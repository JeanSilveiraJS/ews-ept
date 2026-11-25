"""Testes do protótipo.

Execução (na pasta prototipo/):  python -X utf8 test_app.py      ou      python -m pytest test_app.py

Os valores esperados correspondem ao modelo e às bases exportados pelo notebook em ../resultados/prototipo/. Se esses
arquivos forem trocados, atualize as constantes ESPERADO_* depois de conferir o novo modelo. Os testes de interface
usam st.testing.v1.AppTest (sem navegador); os de extração são pulados se os microdados não estiverem em ../dados/.
"""
import json
import re

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import extracao as EX
import nucleo as N

ESPERADO_ARVORES = 262
ESPERADO_PESO_EVADIDOS = 1.135449
ESPERADO_BASE = (1829, {'Baixo': 522, 'Médio': 766, 'Alto': 541})
# Unidade usada nos testes de unidade inteira e de fidelidade da extração (recorte da base de demonstração).
UNIDADE = ('IFMT', 'Campus Cuiabá')
ESPERADO_UNIDADE = (182, {'Baixo': 96, 'Médio': 68, 'Alto': 18})
ESPERADO_DEMO = {'IFCE': 3, 'IFES': 3, 'IFMT': 3, 'IFPA': 3, 'IFSUL': 3}   # instituição: unidades
COLUNAS_PROIBIDAS = {'Categoria da Situação', 'Situação de Matrícula', 'Código da Matricula',
                     'Código do Ciclo Matricula', 'Mês De Ocorrência da Situação', 'Idade', 'Faixa Etária'}
CHAVE_K = ['Instituição', 'Unidade de Ensino', 'Nome de Curso', 'Sexo', 'Tipo de Deficiencia']
APP = str(N.PASTA / 'app.py')


def _esquema():
    return N.carregar_modelo_e_esquema()[1]


def _csv(tabela, sep=',', enc='utf-8'):
    return tabela.to_csv(index=False, sep=sep).encode(enc)


def _base(arquivo=N.DEMO_FILE):
    return N.ler_csv(arquivo.read_bytes())


def _unidade():
    """Matrículas de UNIDADE na base de demonstração, numeradas a partir de 0."""
    base = _base()
    return base[base['Instituição'].eq(UNIDADE[0]) & base['Unidade de Ensino'].eq(UNIDADE[1])].reset_index(drop=True)


def _erro(funcao, *args):
    try:
        funcao(*args)
    except N.ErroImportacao as erro:
        return str(erro)
    raise AssertionError('deveria ter levantado ErroImportacao')


# ---------------------------------------------------------------- modelo, bases e SHAP
def test_modelo_e_esquema():
    modelo, esquema = N.carregar_modelo_e_esquema()
    assert len(esquema['colunas']) == 21 and len(esquema['categoricas']) == 13 and len(esquema['numericas']) == 8
    assert esquema['grupos']['estudante'] == ['Sexo', 'Tipo de Deficiencia', N.IDADE_MODELO]
    excluidos = {'Forma de ingresso', 'Identidade de Gênero', 'Tipo de Necessidade Educacional Específica',
                 'Renda Familiar', 'Cor / Raça'}
    assert not excluidos & set(esquema['colunas']), 'campos de cadastro desigual não devem entrar no modelo'
    assert 'Tipo de Curso' not in esquema['colunas'] and 'Formação de Professores' not in esquema['colunas']
    assert modelo.tree_count_ == ESPERADO_ARVORES, modelo.tree_count_
    assert abs(modelo.get_all_params()['class_weights'][1] - ESPERADO_PESO_EVADIDOS) < 1e-4
    assert esquema['limiares'] == {'medio': 0.3, 'alto': 0.7}
    resultados = N.carregar_resultados()
    assert resultados['modelo']['arvores'] == ESPERADO_ARVORES
    assert round(resultados['uso_na_unidade']['Só cadastro completo']['ROC-AUC dentro da unidade'], 3) == 0.700
    print(f'modelo/esquema OK: 21 atributos (13 categóricos), {ESPERADO_ARVORES} árvores, limiares 0,30/0,70')


def test_bases_e_aditividade_shap():
    """Colunas, ausência de rótulo e identificadores, níveis e aditividade SHAP na base e numa unidade dela."""
    modelo, esquema = N.carregar_modelo_e_esquema()
    for nome, bruto, (n_esperado, niveis_esperados) in ((N.DEMO_FILE.name, _base(), ESPERADO_BASE),
                                                         (' – '.join(UNIDADE), _unidade(), ESPERADO_UNIDADE)):
        assert not COLUNAS_PROIBIDAS & set(bruto.columns), nome
        assert list(bruto.columns) == [N.nome_csv(c) for c in esquema['colunas']], nome
        X, avisos, ids = N.preparar_entradas(bruto, esquema)
        escore = modelo.predict_proba(X)[:, 1]
        shap = N.shap_valores(modelo, X, esquema)
        erro_bruto = float(np.abs(shap.sum(1) - modelo.predict(X, prediction_type='RawFormulaVal')).max())
        erro_escore = float(np.abs(1 / (1 + np.exp(-shap.sum(1))) - escore).max())
        niveis = pd.Series(N.nivel_alerta(escore, .3, .7)).value_counts().to_dict()
        assert len(X) == n_esperado and niveis == niveis_esperados, (nome, niveis)
        assert not avisos and ids is None and erro_bruto < 1e-9 and erro_escore < 1e-9
        print(f'{nome}: {len(X)} registros, níveis {niveis}, |base+ΣSHAP − saída| {erro_bruto:.1e}')


def test_criterios_da_base_de_demonstracao():
    """Cinco institutos (um por região), três unidades cada e grupos de ao menos 5 pessoas (regra de supressão)."""
    base = _base()
    assert base.groupby('Instituição')['Unidade de Ensino'].nunique().to_dict() == ESPERADO_DEMO
    assert base['UF'].nunique() == 5
    assert base.groupby(CHAVE_K, dropna=False).size().min() >= 5
    print(f'base de demonstração OK: {len(base)} registros, 15 unidades, menor grupo '
          f'{base.groupby(CHAVE_K).size().min()} pessoas')


# ---------------------------------------------------------------- importação de CSV
def test_leitura_e_validacao_de_entrada():
    """Idade, CSV com ';' e Latin-1, valores inválidos, coluna faltante, duração por datas e identificação."""
    esquema = _esquema()
    dados = _base().head(5).astype({N.IDADE_CSV: object, 'Carga Horaria': object})
    X1, _, _ = N.preparar_entradas(dados, esquema)
    X2, _, _ = N.preparar_entradas(dados.rename(columns={N.IDADE_CSV: N.IDADE_MODELO}), esquema)
    assert X1.equals(X2), 'nome interno e nome de interface da idade devem dar a mesma entrada'
    dados.loc[0, 'Carga Horaria'] = 'abc'
    dados.loc[1, N.IDADE_CSV] = 140
    X3, avisos, _ = N.preparar_entradas(N.ler_csv(_csv(dados, ';', 'latin-1')), esquema)
    assert X3.shape == (5, 21) and len(avisos) == 2 and np.isnan(X3.loc[0, 'Carga Horaria']), avisos
    msg = _erro(N.preparar_entradas, dados.drop(columns=[N.IDADE_CSV, 'Turno']), esquema)
    assert f'"{N.IDADE_CSV}"' in msg and '"Turno"' in msg and 'Faltam 2' in msg, msg
    com_datas = dados.drop(columns=['Duracao_Ciclo_Dias']).assign(
        **{N.DATAS_CICLO[0]: '01/02/2022', N.DATAS_CICLO[1]: '31/01/2024'})
    assert (N.preparar_entradas(com_datas, esquema)[0]['Duracao_Ciclo_Dias'] == 729).all()
    assert N.DATAS_CICLO[0] in _erro(N.preparar_entradas, dados.drop(columns=['Duracao_Ciclo_Dias']), esquema)
    # Identificação opcional (com grafia diferente): exibida, mas fora das entradas do modelo.
    _, avisos, ids = N.preparar_entradas(dados.assign(identificacao=['a1', 'a2', '', None, 'a5']), esquema)
    assert ids.tolist() == ['a1', 'a2', '', '', 'a5'] and 'grafia diferente' in avisos[0]
    print('entrada OK: idade, ";"/Latin-1, avisos, colunas faltantes nomeadas, datas → duração, identificação')


def test_mensagens_de_erro_de_importacao():
    esquema = _esquema()
    assert 'vazio' in _erro(N.ler_csv, b'') and 'vazio' in _erro(N.ler_csv, b'   \n  ')
    assert 'Excel' in _erro(N.ler_csv, b'PK\x03\x04conteudo') and 'Excel' in _erro(N.ler_csv, b'\xd0\xcf\x11\xe0xx')
    assert 'separador' in _erro(N.ler_csv, b'coluna unica\n1\n2\n')
    assert 'nenhum registro' in _erro(N.ler_csv, b'a,b,c\n')
    assert 'interpretar' in _erro(N.ler_csv, b'a,b,c\n1,2,3\n4,5,6,7,8\n')
    dados = _base().head(3)
    duplicado = pd.concat([dados, dados[['Turno']].rename(columns={'Turno': 'turno'})], axis=1)
    assert 'repetidas' in _erro(N.preparar_entradas, duplicado, esquema)
    cabecalho_repetido = _csv(dados).replace(b'Sexo,', b'Turno,', 1)
    assert '"Turno" e "Turno.1"' in _erro(N.preparar_entradas, N.ler_csv(cabecalho_repetido), esquema)
    print('mensagens de importação OK')


def test_harmonizacao_e_ausentes():
    """Cabeçalhos com grafia diferente e campos vazios (categóricos viram "Desconhecido"; numéricos, ausentes)."""
    modelo, esquema = N.carregar_modelo_e_esquema()
    dados = _base().head(4)
    X_ref, _, _ = N.preparar_entradas(dados, esquema)
    variantes = dados.rename(columns={'Unidade de Ensino': 'unidade de ensino', 'Tipo de Deficiencia':
                                      'Tipo de Deficiência', N.IDADE_CSV: 'IDADE_NO_INGRESSO'})
    X, avisos, _ = N.preparar_entradas(N.ler_csv(_csv(variantes)), esquema)
    assert X.equals(X_ref) and len(avisos) == 1 and 'grafia diferente' in avisos[0], avisos
    faltando = dados.astype(object)
    faltando.loc[0, 'Turno'] = ''
    faltando.loc[1, 'Sexo'] = None
    faltando.loc[2, N.IDADE_CSV] = ''
    X, avisos, _ = N.preparar_entradas(N.ler_csv(_csv(faltando, ';')), esquema)
    assert not avisos, 'campo vazio é ausência, não valor inválido'
    assert X.loc[0, 'Turno'] == 'Desconhecido' and X.loc[1, 'Sexo'] == 'Desconhecido'
    assert np.isnan(X.loc[2, N.IDADE_MODELO]) and np.isfinite(modelo.predict_proba(X)[:, 1]).all()
    assert N.fmt_valor(N.IDADE_MODELO, np.nan) == 'não informado'
    assert N.fmt_valor('Turno', 'Desconhecido') == 'não informado'
    print('harmonização e ausentes OK')


# ---------------------------------------------------------------- lista da unidade e ficha
def test_niveis_de_alerta_nos_limiares():
    niveis = N.nivel_alerta([0.0, 0.2999, 0.30, 0.6999, 0.70, 1.0], 0.3, 0.7).tolist()
    assert niveis == ['Baixo', 'Baixo', 'Médio', 'Médio', 'Alto', 'Alto'], niveis


def test_lista_da_unidade_e_selecao():
    modelo, esquema = N.carregar_modelo_e_esquema()
    X, _, _ = N.preparar_entradas(_base(), esquema)
    uni = N.filtrar(X, 'IFMT', 'Campus Cuiabá')
    assert len(uni) == 182 and uni.index.equals(X.index[(X['Instituição'] == 'IFMT')
                                                        & (X['Unidade de Ensino'] == 'Campus Cuiabá')])
    escore, shap = modelo.predict_proba(uni)[:, 1], N.shap_valores(modelo, uni, esquema)
    tabela = N.tabela_estudantes(uni, escore, N.nivel_alerta(escore, .3, .7), shap, esquema)
    assert tabela['Escore'].is_monotonic_decreasing and tabela['Posição'].tolist() == list(range(1, 183))
    assert set(tabela['Nº']) == set(uni.index + 1), 'Nº deve ser a posição da linha no arquivo'
    linha = tabela.iloc[7]
    assert np.isclose(modelo.predict_proba(X.loc[[linha['Nº'] - 1]])[:, 1][0], linha['Escore'])
    assert linha['Estudante'] == f'Registro {linha["Nº"]}'
    # O fator principal é o atributo de maior contribuição positiva.
    i = uni.index.get_loc(linha['Nº'] - 1)
    coluna = esquema['colunas'][int(np.argmax(shap[i, :-1]))]
    assert linha['Fator principal'].startswith(N.rotulo(coluna) + ':')
    assert not tabela['Curso'].str.startswith('Técnico em').any()
    assert N.registro_selecionado(tabela, []) == tabela['Nº'].iat[0]
    assert N.registro_selecionado(tabela, [3]) == tabela['Nº'].iat[3]
    assert N.registro_selecionado(tabela, [999]) == tabela['Nº'].iat[0]
    assert N.registro_selecionado(tabela.iloc[0:0], [0]) is None
    ids = pd.Series(['m-' + str(k) for k in range(len(X))], index=X.index)
    com_id = N.tabela_estudantes(uni, escore, N.nivel_alerta(escore, .3, .7), shap, esquema, ids)
    assert com_id['Estudante'].iat[0] == f'm-{com_id["Nº"].iat[0] - 1}'
    print('lista da unidade, fator principal e seleção OK')


def test_explicacao_individual():
    modelo, esquema = N.carregar_modelo_e_esquema()
    X, _, _ = N.preparar_entradas(_unidade(), esquema)
    linha = X.loc[[10]]
    sv = N.shap_valores(modelo, linha, esquema)[0]
    exp = N.explicar_registro(linha, sv[:-1], esquema)
    assert len(exp) == 21 and exp['Contribuição'].abs().is_monotonic_decreasing
    assert np.isclose(exp['Contribuição'].sum() + sv[-1], modelo.predict(linha, prediction_type='RawFormulaVal')[0])
    assert ((exp['Contribuição'] >= 0) == (exp['Efeito'] == 'Eleva o escore')).all()
    grupos = N.participacao_grupos(exp)
    assert abs(sum(grupos.values()) - 1) < 1e-9 and set(grupos) == set(N.GRUPOS)
    texto = N.resumo_textual(exp)
    assert exp.iloc[0]['Valor'] in texto and 'uF' not in texto, texto
    assert N.fmt_num(1234.5, 1) == '1.234,5' and N.fmt_num(0.80149, 3) == '0,801'
    assert N.fmt_valor(N.IDADE_MODELO, 30.8) == '30,8 anos' and N.fmt_valor('Carga Horaria', 1620.0) == '1.620 h'
    imp = N.importancia_por_atributo(N.shap_valores(modelo, X, esquema), esquema)
    assert len(imp) == 21 and imp['Importância'].is_monotonic_decreasing
    assert N.auc([0, 0, 1, 1], [.1, .4, .35, .8]) == 0.75 and N.auc([1, 1], [.2, .3]) is None
    print('explicação individual OK:', texto)


def test_modelo_de_csv_reimportavel():
    esquema = _esquema()
    conteudo = N.modelo_csv(esquema, pd.read_csv(N.DEMO_FILE, encoding='utf-8-sig').head(3))
    X, avisos, ids = N.preparar_entradas(N.ler_csv(conteudo), esquema)
    assert X.shape == (3, 21) and not avisos and ids.tolist() == ['exemplo-1', 'exemplo-2', 'exemplo-3']
    print('modelo de CSV OK: reimportado sem avisos')


# ---------------------------------------------------------------- interface (AppTest)
def _metrica(at, rotulo):
    return [m.value for m in at.metric if m.label == rotulo][0]


def _ficha(at):
    """Título da ficha (nome do estudante) e escore exibido na legenda."""
    titulo = [s.value for s in at.subheader if s.value.startswith('Registro ') or s.value.startswith('m-')][0]
    legenda = [c.value for c in at.caption if 'na unidade · escore' in c.value][0]
    return titulo, float(re.search(r'escore (\d+,\d+)', legenda).group(1).replace(',', '.'))


def X_demo():
    return N.preparar_entradas(_base(), _esquema())[0]


def _novo_app():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception and not at.error, (at.exception, at.error)
    return at


def test_interface_painel():
    """Unidade padrão, troca de unidade (limpa a seleção), filtro de nível, seleção e ordem do gráfico."""
    at = _novo_app()
    assert at.title[0].value == 'Campus Iguatu' and _metrica(at, 'Estudantes') == '115'
    assert [_metrica(at, f'Nível {n}') for n in ('alto', 'médio', 'baixo')] == ['23', '59', '33']
    reg0, esc0 = _ficha(at)
    at.session_state['tabela'] = {'selection': {'rows': [5], 'columns': [], 'cells': []}}
    at.run()
    reg5, esc5 = _ficha(at)
    assert not at.exception and reg5 != reg0 and esc5 <= esc0
    tabela = [d.value for d in at.dataframe if 'Fator principal' in d.value.columns][0]
    assert tabela['Estudante'].iat[5] == reg5
    # Regressão: a ordem das barras do gráfico acompanha o estudante selecionado.
    grafico = json.loads(at.get('vega_lite_chart')[0].proto.spec)
    linha = X_demo().loc[[int(reg5.split()[1]) - 1]]
    modelo, esquema = N.carregar_modelo_e_esquema()
    exp = N.explicar_registro(linha, N.shap_valores(modelo, linha, esquema)[0, :-1], esquema)
    assert grafico['encoding']['y']['sort'] == (exp['Atributo'] + ': ' + exp['Valor']).head(6).tolist()

    at.selectbox(key='inst').set_value('IFMT').run()
    assert not at.exception and at.selectbox(key='uni').value == 'Campus Cuiabá'
    assert 'tabela' not in at.session_state or not at.session_state['tabela']['selection']['rows']
    assert _metrica(at, 'Estudantes') == '182' and _metrica(at, 'Nível alto') == '18'
    at.segmented_control(key='nivel').set_value('Alto').run()
    assert not at.exception and len([d.value for d in at.dataframe if 'Fator principal' in d.value.columns][0]) == 18
    at.session_state['tabela'] = {'selection': {'rows': [999], 'columns': [], 'cells': []}}
    at.run()
    assert not at.exception, 'seleção fora do intervalo não deve quebrar o app'
    at.selectbox(key='uni').set_value('Campus Sinop').run()
    at.segmented_control(key='nivel').set_value('Alto').run()
    assert any('Nenhum estudante neste filtro' in i.value for i in at.info)
    print('painel OK: unidades, níveis, seleção e filtro vazio')


def test_interface_envio_de_csv():
    at = _novo_app()
    at.segmented_control(key='fonte').set_value('csv').run()
    assert not at.exception and any('Envie a lista' in i.value for i in at.info) and at.download_button
    dados = _unidade().head(30).astype({'Carga Horaria': object})
    dados.loc[0, 'Carga Horaria'] = 'n/d'
    dados.insert(0, N.COLUNA_ID, [f'm-{i}' for i in range(30)])
    at.file_uploader(key='arquivo').set_value(('ingressantes.csv', _csv(dados, ';', 'latin-1'), 'text/csv')).run()
    assert not at.exception and not at.error, at.error
    assert _metrica(at, 'Estudantes') == '30' and any('não numérico' in w.value for w in at.warning)
    assert _ficha(at)[0].startswith('m-'), 'a identificação do CSV deve aparecer na ficha'
    at.file_uploader(key='arquivo').set_value(('ruim.csv', _csv(dados.drop(columns=['UF'])), 'text/csv')).run()
    assert not at.exception and at.error and '"UF"' in at.error[0].value
    at.file_uploader(key='arquivo').set_value(('planilha.csv', b'PK\x03\x04xx', 'text/csv')).run()
    assert not at.exception and 'Excel' in at.error[0].value
    print('envio de CSV OK: identificação, aviso, coluna faltante e planilha do Excel')


def test_pagina_do_modelo():
    at = _novo_app()
    at.switch_page('app_pages/modelo.py').run()
    assert not at.exception and at.title[0].value == 'Como o modelo decide'
    assert _metrica(at, 'Dentro da unidade, só cadastro completo') == '0,70'
    assert _metrica(at, 'Nível alto') == '78%'
    at.number_input(key='md_registro').set_value(10).run()
    tabelas = [t.value for t in at.table]
    decomposicao = [t for t in tabelas if 'Confere?' in t.index][0]
    assert decomposicao.loc['Confere?'].iat[0] == 'sim'
    print('página do modelo OK')


# ---------------------------------------------------------------- extração dos microdados
def _sem_microdados():
    faltam = [n for n, _, _ in EX.EDICOES.values() if not (EX.PASTA_DADOS / n).exists()]
    if faltam:
        print('PULADO (microdados ausentes em dados/):', faltam)
    return bool(faltam)


def test_extracao_reproduz_base_do_notebook():
    """Fidelidade: UNIDADE, ano-base 2025, turmas de teste, com desfecho e regra dos 5 = recorte da base do notebook."""
    if _sem_microdados():
        return
    esquema = _esquema()
    tabela, ausentes, sha_ok = EX.carregar_tecnicos(2025, esquema)
    assert sha_ok and not ausentes
    mapa = EX.carregar_mapa_particao()
    assert pd.Series(mapa).value_counts().to_dict() == {'Treino': 2882, 'Validação': 618, 'Teste': 618}
    r = tabela[tabela['Instituição'].eq(UNIDADE[0]) & tabela['Unidade de Ensino'].eq(UNIDADE[1])
               & tabela[EX.SITUACAO].isin(['Concluintes', 'Evadidos']) & tabela[EX.CICLO].map(mapa).eq('Teste')]
    dados, notas = EX.para_formato_prototipo(r, 2025, esquema)
    X_extraido, _, _ = N.preparar_entradas(dados, esquema)
    X_extraido = X_extraido[X_extraido.groupby(CHAVE_K)['UF'].transform('size').ge(5)].reset_index(drop=True)
    X_base, _, _ = N.preparar_entradas(_unidade(), esquema)
    assert not notas and X_extraido.equals(X_base), 'extração ≠ base do notebook'
    resumo = EX.resumo_ciclos(r, 2025, mapa)
    assert resumo['Matrículas'].sum() == 186 and set(resumo['Uso no modelo']) == {EX.USO_NO_MODELO['Teste']}
    print(f'extração 2025 OK: reproduz {" – ".join(UNIDADE)} da base de demonstração (186 matrículas extraídas, '
          f'{len(X_base)} após a regra dos 5, {len(resumo)} turmas de teste)')


def test_extracao_edicao_2024():
    """Edição 2024: turmas fora do treino, idade em 31/12/2024, programa associado derivado e tipo de deficiência
    suposto ("Não possui deficiência"), sinalizado na nota."""
    if _sem_microdados():
        return
    modelo, esquema = N.carregar_modelo_e_esquema()
    tabela, ausentes, sha_ok = EX.carregar_tecnicos(2024, esquema)
    assert sha_ok and sorted(ausentes) == ['Programa Associado', 'Tipo de Deficiencia'], ausentes
    assert not tabela[EX.CICLO].isin(EX.carregar_mapa_particao()).any(), 'nenhuma turma de 2024 está no treino'
    r = tabela[tabela['Instituição'].eq('IFPA') & tabela['Unidade de Ensino'].eq('Campus Tucuruí')]
    resumo = EX.resumo_ciclos(r, 2024, {})
    assert resumo['Matrículas'].sum() == len(r) == resumo[['Concluintes', 'Evadidos', 'Em curso']].sum().sum()
    dados, notas = EX.para_formato_prototipo(r, 2024, esquema)
    assert notas == [EX.NOTA_2024]
    proeja = dados['Tipo de Oferta'].fillna('').str.startswith('PROEJA')
    assert (dados['Programa Associado'] == np.where(proeja, 'PROEJA', 'Sem Programa Associado')).all()
    linha = r.iloc[0]
    inicio = pd.to_datetime(linha[EX.DATAS[0]], dayfirst=True)
    assert dados['Idade no ingresso'].iat[0] == round(float(linha['Idade']) - (pd.Timestamp('2024-12-31')
                                                                              - inicio).days / 365.25, 1)
    X, _, _ = N.preparar_entradas(dados, esquema)
    assert X.shape == (len(r), 21) and np.isfinite(modelo.predict_proba(X)[:, 1]).all()
    assert X[EX.DEFICIENCIA].eq(EX.DEFICIENCIA_SUPOSTA).all()
    com_situacao, _ = EX.para_formato_prototipo(r, 2024, esquema, incluir_situacao=True)
    assert list(com_situacao.columns)[-1] == EX.SITUACAO and N.preparar_entradas(com_situacao, esquema)[0].equals(X)
    print(f'extração 2024 OK: {len(r)} matrículas de {len(resumo)} turmas do IFPA – Campus Tucuruí')


def test_pagina_extracao_e_envio_ao_painel():
    """AppTest: extração (2025) → seleção de turmas → "Abrir no painel" → painel com o recorte."""
    if _sem_microdados():
        return
    at = _novo_app()
    at.switch_page('app_pages/extrair.py').run()
    assert not at.exception and any('Escolha a instituição' in i.value for i in at.info)
    at.selectbox(key='ex_inst').set_value('IFES').run()
    at.selectbox(key='ex_uni').set_value('Campus Alegre').run()
    assert not at.exception and any('Marque uma ou mais turmas' in i.value for i in at.info)
    ciclos = [d.value for d in at.dataframe][0]
    teste = ciclos.index[ciclos['Uso no modelo'].eq(EX.USO_NO_MODELO['Teste'])][:2].tolist()
    at.session_state['ex_ciclos'] = {'selection': {'rows': teste, 'columns': [], 'cells': []}}
    at.run()
    assert not at.exception, at.exception
    n = int(ciclos.loc[teste, 'Matrículas'].sum())
    assert _metrica(at, 'Matrículas') == N.fmt_num(n, 0) and _metrica(at, 'Turmas') == '2'
    cruzada = [d.value for d in at.dataframe if 'Evadiram (entre os com desfecho)' in d.value.columns][0]
    assert cruzada[['Concluintes', 'Evadidos', 'Em curso']].to_numpy().sum() == n
    assert _metrica(at, 'ROC-AUC no recorte') != '—'
    assert not any('treinamento' in w.value for w in at.warning)
    at.button(key='ex_abrir').click().run()
    assert not at.exception, at.exception
    assert at.title[0].value == 'Campus Alegre' and _metrica(at, 'Estudantes') == N.fmt_num(n, 0)
    assert at.segmented_control(key='fonte').value == 'extraido'
    at.switch_page('app_pages/painel.py')   # o AppTest volta à página anterior após st.switch_page
    at.segmented_control(key='fonte').set_value('demo').run()
    assert not at.exception and _metrica(at, 'Estudantes') == '115'
    print(f'extração → painel OK: 2 turmas de teste ({n} matrículas)')


TESTES = [v for k, v in dict(globals()).items() if k.startswith('test_')]

if __name__ == '__main__':
    for teste in TESTES:
        teste()
    print(f'OK: todos os {len(TESTES)} testes passaram')
