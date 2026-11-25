"""Extração de recortes dos microdados da PNP para a demonstração do protótipo.

Escolhe-se a edição (ano-base 2024 ou 2025), a instituição, a unidade, o curso e um ou mais ciclos (turmas). As
matrículas são convertidas para o formato de entrada do painel com as mesmas derivações do notebook: duração do ciclo
pelas datas e idade no ingresso estimada a partir da coluna Idade (referida a 31/12 do ano-base). O restante do
preparo (categorias vazias, números com vírgula) fica a cargo de nucleo.preparar_entradas, como em qualquer CSV.

A edição 2024 não tem duas colunas usadas pelo modelo. Programa Associado descreve a oferta e é derivado do tipo de
oferta (coincide com o valor real em todo o teste de 2025). Tipo de Deficiencia é individual: assume-se "Não possui
deficiência", valor de 98,4% das matrículas técnicas, e a suposição é sinalizada (ver NOTA_2024).
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from nucleo import MAPA_PARTICAO_FILE

PASTA_DADOS = Path(__file__).resolve().parent.parent / 'dados'
EDICOES = {  # ano-base: arquivo, SHA-256 documentado em dados/LEIA-ME.md e descrição
    2025: ('microdados_eficiencia_academica_2025.csv.gz',
           '18e9ad28b486c4793ce41f92e7b113c8a931aff522673707614db35687f00bed',
           'Edição usada no treino e na avaliação. Só as turmas do teste são novas para o modelo; turmas de treino e '
           'validação recebem escores otimistas.'),
    2024: ('microdados_eficiencia_academica_2024.csv.gz',
           'f8bdd8aa71b6be2cd05e00e3a991a119a381d18b210650cfa8d87a7c440763cd',
           'Edição anterior, com turmas encerradas em 2023 que o modelo nunca viu. Não traz duas colunas usadas '
           'pelo modelo; serve para mostrar o processamento de outra edição, não como validação.'),
}
CICLO, SITUACAO = 'Código do Ciclo Matricula', 'Categoria da Situação'
DATAS = ('Data de Inicio do Ciclo', 'Data de Fim Previsto do Ciclo')
DERIVADOS = ('Duracao_Ciclo_Dias', 'Idade_Inicio_Estimada')
DEFICIENCIA = 'Tipo de Deficiencia'
DEFICIENCIA_SUPOSTA = 'Não possui deficiência'
USO_NO_MODELO = {'Treino': 'Treino (escore otimista)', 'Validação': 'Validação (parada do treino)',
                 'Teste': 'Teste (turma nova para o modelo)', None: 'Não usada no treinamento'}
# Medido no teste de 2025: supor "Não possui deficiência" para todos muda o nível de 0,7% das matrículas; deixar o
# campo vazio (valor que o modelo nunca viu) mudaria 34% e baixaria o escore médio em 0,11.
NOTA_2024 = ('O tipo de deficiência não existe nesta edição: supôs-se "Não possui deficiência" para todos (valor de '
             '98,4% das matrículas técnicas). No teste de 2025, essa suposição muda o nível de 0,7% das matrículas; '
             'deixar o campo vazio mudaria 34%. O programa associado foi derivado do tipo de oferta.')


def colunas_necessarias(esquema):
    brutas = [c for c in esquema['colunas'] if c not in DERIVADOS]
    return list(dict.fromkeys(brutas + ['Idade', *DATAS, CICLO, SITUACAO, 'Tipo de Curso']))


def sha256(caminho):
    resumo = hashlib.sha256()
    with open(caminho, 'rb') as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b''):
            resumo.update(bloco)
    return resumo.hexdigest()


def carregar_tecnicos(ano, esquema, pasta=PASTA_DADOS):
    """Matrículas de cursos técnicos de uma edição (todas as situações), como texto, na ordem do arquivo.

    Retorna (tabela, colunas do esquema ausentes no arquivo, SHA-256 confere?).
    """
    nome, sha_documentado, _ = EDICOES[ano]
    caminho = Path(pasta) / nome
    if not caminho.exists():
        raise FileNotFoundError(caminho)
    cabecalho = pd.read_csv(caminho, sep=';', nrows=0, compression='gzip').columns
    usar = [c for c in colunas_necessarias(esquema) if c in cabecalho]
    ausentes = [c for c in esquema['colunas'] if c not in DERIVADOS and c not in cabecalho]
    tabela = pd.read_csv(caminho, sep=';', compression='gzip', usecols=usar, dtype=str)
    tabela = tabela[tabela['Tipo de Curso'].eq('Técnico')].reset_index(drop=True)
    return tabela, ausentes, sha256(caminho) == sha_documentado


def carregar_mapa_particao(arquivo=MAPA_PARTICAO_FILE):
    """Ciclo → partição do treinamento (Treino/Validação/Teste); dicionário vazio se o arquivo não existir."""
    if not Path(arquivo).exists():
        return {}
    mapa = pd.read_csv(arquivo, dtype=str, encoding='utf-8-sig')
    return dict(zip(mapa[CICLO], mapa['particao']))


def resumo_ciclos(registros: pd.DataFrame, ano, mapa_particao: dict) -> pd.DataFrame:
    """Uma linha por ciclo (turma): curso, datas, oferta, turno, matrículas por situação e uso no modelo."""
    inicio = pd.to_datetime(registros[DATAS[0]], dayfirst=True, errors='coerce')
    fim = pd.to_datetime(registros[DATAS[1]], dayfirst=True, errors='coerce')
    base = registros.assign(_inicio=inicio, _fim=fim)
    resumo = base.groupby(CICLO, sort=False).agg(
        Curso=('Nome de Curso', 'first'), Início=('_inicio', 'min'), **{'Fim previsto': ('_fim', 'max')},
        Oferta=('Tipo de Oferta', 'first'), Turno=('Turno', 'first'), Matrículas=(CICLO, 'size'))
    situacoes = pd.crosstab(base[CICLO], base[SITUACAO])
    for coluna in ('Concluintes', 'Evadidos', 'Em curso'):
        resumo[coluna] = (situacoes.get(coluna, pd.Series(0, index=situacoes.index))
                          .reindex(resumo.index).fillna(0).astype(int))
    resumo['Uso no modelo'] = [USO_NO_MODELO[mapa_particao.get(c) if ano == 2025 else None] for c in resumo.index]
    resumo = resumo.reset_index().rename(columns={CICLO: 'Ciclo'})
    resumo['Curso'] = resumo['Curso'].str.replace(r'^Técnico em ', '', regex=True)
    return resumo.sort_values(['Início', 'Curso', 'Ciclo'], kind='stable').reset_index(drop=True)


def para_formato_prototipo(registros: pd.DataFrame, ano, esquema, incluir_situacao=False):
    """Converte matrículas dos microdados para as colunas do CSV do painel. Retorna (tabela, notas)."""
    notas = []
    inicio = pd.to_datetime(registros[DATAS[0]], dayfirst=True, errors='coerce')
    fim = pd.to_datetime(registros[DATAS[1]], dayfirst=True, errors='coerce')
    saida = registros.copy()
    saida['Duracao_Ciclo_Dias'] = (fim - inicio).dt.days
    idade = pd.to_numeric(registros['Idade'], errors='coerce')
    idade = idade.where(idade.between(5, 100))                      # como no notebook: valor implausível → ausente
    referencia = pd.Timestamp(f'{ano}-12-31')                       # Idade é medida em 31/12 do ano-base
    saida['Idade no ingresso'] = (idade - (referencia - inicio).dt.days / 365.25).round(1)
    if 'Programa Associado' not in saida.columns:
        oferta = saida['Tipo de Oferta'].fillna('')
        saida['Programa Associado'] = np.where(oferta.str.startswith('PROEJA'), 'PROEJA', 'Sem Programa Associado')
    if DEFICIENCIA not in saida.columns:
        saida[DEFICIENCIA] = DEFICIENCIA_SUPOSTA
        notas.append(NOTA_2024)
    colunas = ['Idade no ingresso' if c == 'Idade_Inicio_Estimada' else c for c in esquema['colunas']]
    # Turma e datas vão junto (o painel só as exibe, na ficha do estudante).
    tabela = saida.reindex(columns=colunas + [CICLO, *DATAS] + ([SITUACAO] if incluir_situacao else []))
    return tabela.reset_index(drop=True), notas
