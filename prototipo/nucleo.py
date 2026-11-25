"""Lógica do protótipo, sem interface: modelo, leitura de CSV, preparo dos dados, níveis e explicações.

Tudo aqui é testado em test_app.py. As páginas (app_pages/) só montam a interface sobre estas funções.
"""
import csv
import io
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

PASTA = Path(__file__).resolve().parent
# Modelo e bases são lidos direto da saída do notebook (seção 10), sem cópia para esta pasta.
PASTA_ARTEFATOS = PASTA.parent / 'resultados' / 'prototipo'
MODELO_FILE = PASTA_ARTEFATOS / 'modelo_evasao.cbm'
ESQUEMA_FILE = PASTA_ARTEFATOS / 'esquema_modelo.json'
RESULTADOS_FILE = PASTA_ARTEFATOS / 'resultados_modelo.json'   # números da avaliação, exportados pelo notebook
DEMO_FILE = PASTA_ARTEFATOS / 'base_demonstracao.csv'          # 15 unidades de 5 institutos (teste do modelo)
MAPA_PARTICAO_FILE = PASTA_ARTEFATOS / 'particao_ciclos_2025.csv'

NIVEIS = ('Alto', 'Médio', 'Baixo')
TODOS = 'Todos'
IDADE_MODELO, IDADE_CSV = 'Idade_Inicio_Estimada', 'Idade no ingresso'
DATAS_CICLO = ('Data de Inicio do Ciclo', 'Data de Fim Previsto do Ciclo')
COLUNA_ID = 'Identificação'   # coluna opcional do CSV (matrícula ou nome): exibida, nunca usada pelo modelo
COLUNA_CICLO = 'Código do Ciclo Matricula'   # coluna opcional (turma), como nos microdados: só exibida

ROTULOS = {  # nomes de exibição; o CSV continua usando os nomes do esquema
    IDADE_MODELO: 'Idade no ingresso', 'Duracao_Ciclo_Dias': 'Duração prevista do ciclo',
    'Tipo de Deficiencia': 'Tipo de deficiência',
    'Eixo Tecnológico': 'Eixo tecnológico', 'Subeixo Tecnológico': 'Subeixo tecnológico', 'Nome de Curso': 'Curso',
    'Tipo de Oferta': 'Tipo de oferta', 'Modalidade de Ensino': 'Modalidade',
    'Fonte de Financiamento': 'Fonte de financiamento', 'Programa Associado': 'Programa associado',
    'Unidade de Ensino': 'Unidade de ensino', 'Carga Horaria': 'Carga horária',
    'Carga Horaria Mínima': 'Carga horária mínima', 'Fator Esforço Curso': 'Fator de esforço do curso',
    'Total de Inscritos': 'Inscritos no processo seletivo', 'Vagas Ofertadas': 'Vagas ofertadas',
    'Vagas Regulares AC': 'Vagas de ampla concorrência',
}
UNIDADES = {IDADE_MODELO: ' anos', 'Duracao_Ciclo_Dias': ' dias', 'Carga Horaria': ' h', 'Carga Horaria Mínima': ' h'}
GRUPOS = {'estudante': 'Estudante', 'curso': 'Curso e oferta', 'instituicao': 'Instituição'}

class ErroImportacao(ValueError):
    """Problema no arquivo enviado, com mensagem destinada ao usuário."""


# ---------------------------------------------------------------- formatação
def rotulo(coluna):
    return ROTULOS.get(coluna, coluna)


def nome_csv(coluna):
    """Nome que a coluna deve ter no CSV."""
    return IDADE_CSV if coluna == IDADE_MODELO else coluna


def fmt_num(valor, casas=3):
    """Número no padrão brasileiro: vírgula decimal e ponto de milhar."""
    return f'{valor:,.{casas}f}'.replace(',', '§').replace('.', ',').replace('§', '.')


def fmt_valor(coluna, valor):
    """Valor de um atributo, legível para quem não conhece os códigos da PNP."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return 'não informado'
    if isinstance(valor, (int, float, np.integer, np.floating)):
        valor = float(valor)
        casas = 1 if coluna == IDADE_MODELO else (3 if coluna == 'Fator Esforço Curso' else 0)
        if casas == 0 and not valor.is_integer():
            casas = 1
        return fmt_num(valor, casas) + UNIDADES.get(coluna, '')
    return 'não informado' if valor == 'Desconhecido' else str(valor)


# ---------------------------------------------------------------- modelo
def carregar_modelo_e_esquema(modelo_file=MODELO_FILE, esquema_file=ESQUEMA_FILE):
    with open(esquema_file, encoding='utf-8') as f:
        esquema = json.load(f)
    modelo = CatBoostClassifier()
    modelo.load_model(str(modelo_file))
    # Impede usar o esquema de um modelo com os pesos de outro.
    if list(modelo.feature_names_) != esquema['colunas']:
        raise ValueError('As colunas do esquema_modelo.json não coincidem com as do modelo .cbm.')
    if [modelo.feature_names_[i] for i in modelo.get_cat_feature_indices()] != esquema['categoricas']:
        raise ValueError('Os atributos categóricos do esquema não coincidem com os do modelo .cbm.')
    return modelo, esquema


def carregar_resultados(arquivo=RESULTADOS_FILE):
    """Números da avaliação gravados pelo notebook; None se o arquivo não existir."""
    if not Path(arquivo).exists():
        return None
    with open(arquivo, encoding='utf-8') as f:
        return json.load(f)


def grupo_atributo(coluna, esquema):
    for grupo, colunas in esquema['grupos'].items():
        if coluna in colunas:
            return grupo
    return 'curso'


def nivel_alerta(escores, lim_medio, lim_alto):
    escores = np.asarray(escores, dtype=float)
    return np.where(escores < lim_medio, 'Baixo', np.where(escores < lim_alto, 'Médio', 'Alto'))


def shap_valores(modelo, X, esquema):
    """Matriz SHAP do CatBoost (escala log-odds). A última coluna é o valor-base."""
    return modelo.get_feature_importance(data=Pool(X, cat_features=esquema['categoricas']), type='ShapValues')


# ---------------------------------------------------------------- leitura do CSV
def _detectar_separador(texto):
    try:
        return csv.Sniffer().sniff(texto[:20000], delimiters=',;\t|').delimiter
    except csv.Error:
        return ','


def ler_csv(conteudo: bytes) -> pd.DataFrame:
    """CSV com separador detectado (',', ';', tabulação ou '|') e codificação UTF-8 (com ou sem BOM) ou Latin-1.

    Levanta ErroImportacao com mensagem compreensível para arquivo vazio, planilha do Excel, separador não
    reconhecido ou linhas malformadas.
    """
    if not conteudo or not conteudo.strip():
        raise ErroImportacao('O arquivo está vazio.')
    if conteudo[:4] in (b'PK\x03\x04', b'\xd0\xcf\x11\xe0'):
        raise ErroImportacao('O arquivo parece ser uma planilha do Excel (.xlsx ou .xls), não um CSV. No Excel, use '
                             '"Salvar como" → "CSV UTF-8 (delimitado por vírgulas)" e envie o arquivo gerado.')
    try:
        texto = conteudo.decode('utf-8-sig')
    except UnicodeDecodeError:
        texto = conteudo.decode('latin-1')
    try:
        dados = pd.read_csv(io.StringIO(texto), sep=_detectar_separador(texto), dtype=str)
    except pd.errors.EmptyDataError:
        raise ErroImportacao('O arquivo não contém dados.') from None
    except (pd.errors.ParserError, csv.Error) as erro:
        raise ErroImportacao('Não foi possível interpretar o CSV: há linhas com número de campos diferente do '
                             f'cabeçalho ou aspas não fechadas. Detalhe técnico: {erro}') from None
    dados.columns = [str(c).strip() for c in dados.columns]
    if dados.shape[1] == 1:
        raise ErroImportacao('Só uma coluna foi identificada. Verifique se o separador é vírgula (,) ou ponto e '
                             'vírgula (;) e se a primeira linha contém os nomes das colunas.')
    if dados.empty:
        raise ErroImportacao('O arquivo tem cabeçalho, mas nenhum registro.')
    return dados


def _normalizar_nome(nome):
    sem_acento = unicodedata.normalize('NFKD', str(nome)).encode('ascii', 'ignore').decode()
    return re.sub(r'[\s_]+', ' ', sem_acento).strip().casefold()


def harmonizar_colunas(dados: pd.DataFrame, esquema: dict):
    """Renomeia colunas que diferem do esquema só por maiúsculas, acentos, espaços ou '_'. Retorna (dados, avisos)."""
    esperados = [nome_csv(c) for c in esquema['colunas']] + [IDADE_MODELO, COLUNA_ID, COLUNA_CICLO, *DATAS_CICLO]
    por_chave = {_normalizar_nome(c): c for c in esperados}
    # Duas colunas para o mesmo atributo ("Turno" e "turno", ou cabeçalho repetido, que o pandas renomeia para
    # "Turno.1") tornam a entrada ambígua.
    candidatas = {}
    for coluna in dados.columns:
        nome = str(coluna)
        original = re.sub(r'\.\d+$', '', nome)
        chave = _normalizar_nome(original if original != nome and original in dados.columns else nome)
        if chave in por_chave:
            candidatas.setdefault(por_chave[chave], []).append(nome)
    duplicadas = [nomes for nomes in candidatas.values() if len(nomes) > 1]
    if duplicadas:
        raise ErroImportacao('O arquivo tem colunas repetidas para o mesmo atributo: '
                             + '; '.join(' e '.join(f'"{n}"' for n in nomes) for nomes in duplicadas)
                             + '. Mantenha apenas uma de cada.')
    renomear = {}
    for coluna in dados.columns:
        alvo = por_chave.get(_normalizar_nome(coluna))
        if alvo and alvo != coluna and alvo not in dados.columns and alvo not in renomear.values():
            renomear[coluna] = alvo
    avisos = []
    if renomear:
        dados = dados.rename(columns=renomear)
        avisos.append('Colunas reconhecidas com grafia diferente: '
                      + '; '.join(f'"{a}" → "{b}"' for a, b in renomear.items()) + '.')
    return dados, avisos


def preparar_entradas(dados: pd.DataFrame, esquema: dict):
    """Aplica aos dados as mesmas transformações do treinamento. Retorna (X, avisos, identificacao).

    O índice de X é a posição de cada linha no arquivo (0, 1, ...), usada como número do estudante. identificacao é a
    coluna opcional COLUNA_ID (texto) ou None.
    """
    dados, avisos = harmonizar_colunas(dados.copy(), esquema)
    dados = dados.reset_index(drop=True)
    if 'Duracao_Ciclo_Dias' not in dados.columns and all(c in dados.columns for c in DATAS_CICLO):
        inicio = pd.to_datetime(dados[DATAS_CICLO[0]], errors='coerce', dayfirst=True)
        fim = pd.to_datetime(dados[DATAS_CICLO[1]], errors='coerce', dayfirst=True)
        dados['Duracao_Ciclo_Dias'] = (fim - inicio).dt.days
    if IDADE_MODELO not in dados.columns and IDADE_CSV in dados.columns:   # na interface, a idade observada
        dados = dados.rename(columns={IDADE_CSV: IDADE_MODELO})
    faltam = [c for c in esquema['colunas'] if c not in dados.columns]
    if faltam:
        dica = (f' A coluna "Duracao_Ciclo_Dias" pode ser substituída por "{DATAS_CICLO[0]}" e "{DATAS_CICLO[1]}".'
                if 'Duracao_Ciclo_Dias' in faltam else '')
        raise ErroImportacao(f'Faltam {len(faltam)} coluna(s) obrigatória(s): '
                             + ', '.join(f'"{nome_csv(c)}"' for c in faltam)
                             + '. Use os nomes do modelo de CSV disponível na barra lateral.' + dica)
    X = dados[esquema['colunas']].copy()
    for c in esquema['categoricas']:
        X[c] = X[c].astype('string').str.strip().replace('', pd.NA).fillna('Desconhecido').astype(str)
    for c in esquema['numericas']:
        original = X[c]
        X[c] = pd.to_numeric(original.astype('string').str.strip().str.replace(',', '.', regex=False),
                             errors='coerce').astype(float)
        invalidos = int((X[c].isna() & original.notna() & original.astype('string').str.strip().ne('')).sum())
        if invalidos:
            avisos.append(f'{invalidos} valor(es) não numérico(s) em "{nome_csv(c)}" foram tratados como não '
                          'informados.')
    fora = int((~X[IDADE_MODELO].between(10, 90) & X[IDADE_MODELO].notna()).sum())
    if fora:
        avisos.append(f'{fora} registro(s) com "{IDADE_CSV}" fora de 10 a 90 anos; verifique os dados.')
    identificacao = (dados[COLUNA_ID].astype('string').fillna('').str.strip()
                     if COLUNA_ID in dados.columns else None)
    return X, avisos, identificacao


def detalhes_ciclo(dados: pd.DataFrame, esquema: dict) -> pd.DataFrame:
    """Turma e datas do ciclo, quando o arquivo as traz (colunas opcionais, nunca usadas pelo modelo).

    O índice é a posição da linha no arquivo, como em preparar_entradas; sem essas colunas, a tabela não tem colunas.
    """
    dados, _ = harmonizar_colunas(dados, esquema)
    dados = dados.reset_index(drop=True)
    saida = pd.DataFrame(index=dados.index)
    if COLUNA_CICLO in dados.columns:
        saida['Turma (código do ciclo)'] = dados[COLUNA_CICLO].astype('string').str.strip()
    for coluna, nome in zip(DATAS_CICLO, ('Início do ciclo', 'Fim previsto do ciclo')):
        if coluna in dados.columns:
            saida[nome] = pd.to_datetime(dados[coluna], errors='coerce', dayfirst=True).dt.strftime('%d/%m/%Y')
    return saida.fillna('não informado')


def modelo_csv(esquema: dict, exemplos: pd.DataFrame | None = None) -> bytes:
    """CSV-modelo com a coluna opcional de identificação, os nomes de coluna esperados e linhas de exemplo."""
    colunas = [nome_csv(c) for c in esquema['colunas']]
    if exemplos is None:
        tabela = pd.DataFrame(columns=[COLUNA_ID] + colunas)
    else:
        tabela = exemplos[colunas].copy()
        tabela.insert(0, COLUNA_ID, [f'exemplo-{i + 1}' for i in range(len(tabela))])
    return tabela.to_csv(index=False).encode('utf-8-sig')


# ---------------------------------------------------------------- análise de uma unidade
def filtrar(X: pd.DataFrame, instituicao=None, unidade=None, curso=TODOS) -> pd.DataFrame:
    """Matrículas de uma instituição, unidade e curso, preservando o índice (número do estudante)."""
    mascara = pd.Series(True, index=X.index)
    for coluna, valor in (('Instituição', instituicao), ('Unidade de Ensino', unidade), ('Nome de Curso', curso)):
        if valor not in (None, TODOS):
            mascara &= X[coluna].eq(valor)
    return X[mascara]


def explicar_registro(linha: pd.DataFrame, contrib, esquema: dict) -> pd.DataFrame:
    """Contribuições SHAP de um estudante, da maior para a menor em valor absoluto, com valores legíveis."""
    colunas = esquema['colunas']
    tabela = pd.DataFrame({'coluna': colunas,
                           'Atributo': [rotulo(c) for c in colunas],
                           'Valor': [fmt_valor(c, linha[c].iat[0]) for c in colunas],
                           'Contribuição': np.asarray(contrib, dtype=float)})
    tabela['Efeito'] = np.where(tabela['Contribuição'] >= 0, 'Eleva o escore', 'Reduz o escore')
    tabela['Grupo'] = [grupo_atributo(c, esquema) for c in colunas]
    ordem = tabela['Contribuição'].abs().sort_values(ascending=False, kind='stable').index
    return tabela.loc[ordem].reset_index(drop=True)


def fator_principal(linha: pd.Series, contrib, esquema: dict) -> str:
    """Atributo que mais elevou o escore, com seu valor ("Forma de ingresso: não informado"); "—" se nenhum."""
    contrib = np.asarray(contrib, dtype=float)
    i = int(np.argmax(contrib))
    if contrib[i] <= 0:
        return '—'
    coluna = esquema['colunas'][i]
    return f'{rotulo(coluna)}: {fmt_valor(coluna, linha[coluna])}'


def tabela_estudantes(X: pd.DataFrame, escore, nivel, shap, esquema: dict, identificacao=None) -> pd.DataFrame:
    """Uma linha por estudante, ordenada pelo escore (maior primeiro).

    Nº é a posição da linha no arquivo (1 = primeira linha de dados) e não muda com filtros; "Posição" é a ordem
    pelo escore. "Estudante" mostra a identificação do CSV, quando houver, ou "Registro <Nº>".
    """
    numeros = X.index.to_numpy() + 1
    ids = identificacao.reindex(X.index).fillna('').to_numpy() if identificacao is not None else [''] * len(X)
    tabela = pd.DataFrame({
        'Nº': numeros,
        'Estudante': [i if i else f'Registro {n}' for i, n in zip(ids, numeros)],
        'Nível': nivel,
        'Escore': np.asarray(escore, dtype=float),
        'Curso': X['Nome de Curso'].str.replace(r'^Técnico em ', '', regex=True).to_numpy(),
        'Oferta': X['Tipo de Oferta'].to_numpy(),
        'Turno': X['Turno'].replace('Desconhecido', '—').to_numpy(),
        'Idade': X[IDADE_MODELO].to_numpy(),
        'Fator principal': [fator_principal(X.iloc[i], shap[i, :-1], esquema) for i in range(len(X))],
    })
    tabela = tabela.sort_values(['Escore', 'Nº'], ascending=[False, True], kind='stable').reset_index(drop=True)
    tabela.insert(0, 'Posição', np.arange(1, len(tabela) + 1))
    return tabela


def registro_selecionado(visivel: pd.DataFrame, linhas) -> int | None:
    """Nº do estudante escolhido na tabela; sem seleção válida, o primeiro da lista. None se a tabela estiver vazia."""
    if visivel.empty:
        return None
    pos = linhas[0] if linhas and 0 <= linhas[0] < len(visivel) else 0
    return int(visivel.iloc[pos]['Nº'])


def resumo_textual(explicacao: pd.DataFrame, n=3) -> str:
    """Frase com os atributos que mais elevaram e mais reduziram o escore."""
    def lista(parte):
        itens = [f'{a if a.isupper() else a[0].lower() + a[1:]} ({v})'
                 for a, v in zip(parte['Atributo'].head(n), parte['Valor'].head(n))]
        return ', '.join(itens[:-1]) + ' e ' + itens[-1] if len(itens) > 1 else itens[0]
    eleva = explicacao[explicacao['Contribuição'] > 0]
    reduz = explicacao[explicacao['Contribuição'] < 0]
    partes = []
    if not eleva.empty:
        partes.append(f'**Elevaram** o escore: {lista(eleva)}.')
    if not reduz.empty:
        partes.append(f'**Reduziram**: {lista(reduz)}.')
    return ' '.join(partes) or 'Nenhum atributo alterou o escore em relação à referência do modelo.'


def participacao_grupos(explicacao: pd.DataFrame) -> dict:
    """Parcela (0–1) da soma dos |SHAP| de cada grupo de atributos (estudante, curso e oferta, instituição)."""
    absoluto = explicacao['Contribuição'].abs()
    total = float(absoluto.sum()) or 1.0
    return {g: float(absoluto[explicacao['Grupo'] == g].sum()) / total for g in GRUPOS}


def importancia_por_atributo(shap, esquema: dict) -> pd.DataFrame:
    """Média de |SHAP| por atributo (importância global), com o grupo de cada um, em ordem decrescente."""
    tabela = pd.DataFrame({'Atributo': [rotulo(c) for c in esquema['colunas']],
                           'Grupo': [GRUPOS[grupo_atributo(c, esquema)] for c in esquema['colunas']],
                           'Importância': np.abs(np.asarray(shap)[:, :-1]).mean(axis=0)})
    return tabela.sort_values('Importância', ascending=False, kind='stable').reset_index(drop=True)


def auc(rotulos, escores) -> float | None:
    """ROC-AUC (probabilidade de um evadido ter escore maior que um concluinte); None sem as duas classes."""
    rotulos, escores = np.asarray(rotulos, dtype=bool), np.asarray(escores, dtype=float)
    pos, neg = escores[rotulos], escores[~rotulos]
    if not len(pos) or not len(neg):
        return None
    ordem = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()   # postos médios tratam empates
    return float((ordem[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
