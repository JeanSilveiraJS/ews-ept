# Early Warning System para evasão em cursos técnicos da Rede Federal

Trabalho de Conclusão de Curso de Tecnologia em Sistemas para Internet (Colégio Politécnico/UFSM).

Prova de conceito de um sistema de alerta precoce de evasão para equipes locais (por unidade de ensino):
- **Modelo:** classificador CatBoost treinado com os Microdados de Eficiência Acadêmica da Plataforma Nilo Peçanha
  (ano-base 2025), com 21 atributos do estudante, do curso e da instituição conhecidos no ingresso.
- **Explicações:** valores SHAP mostram quais atributos mais pesaram no escore de cada estudante.
- **Protótipo:** painel Streamlit local em que a equipe de uma unidade vê, no início de cada ciclo, os estudantes
  ordenados pelo escore, com nível de alerta e explicação de cada caso.
- **Artigo:** texto do TCC em LaTeX (modelo UFSM), mantido fora deste repositório. Os números, as figuras e as
  capturas do artigo vêm do notebook e do protótipo daqui.

## Estrutura

```
ews-ept/
├── modelagem_evasao.ipynb   notebook: dados → modelo → avaliação → SHAP → exportação (roda no Google Colab)
├── requirements.txt         dependências do notebook
├── dados/                   microdados da PNP (não versionados; origem, download e SHA-256 no LEIA-ME)
├── resultados/              saídas do notebook
│   ├── figuras/             gráficos (três deles usados no artigo)
│   ├── tabelas/             tabelas em CSV
│   ├── resultados.json      números da avaliação
│   └── prototipo/           modelo, esquema e bases de demonstração, lidos diretamente pelo protótipo
└── prototipo/               aplicação Streamlit e testes
```

O fluxo é de mão única: `dados/` → notebook → `resultados/` → protótipo. O protótipo não tem cópia própria do
modelo; ele lê `resultados/prototipo/`.

As decisões do modelo (população, atributos excluídos, idade estimada, divisão por turma, referências de comparação,
níveis) estão explicadas no próprio notebook, cada uma na sua seção.

## Executar o notebook

**Google Colab:** abra
[modelagem_evasao.ipynb no Colab](https://colab.research.google.com/github/JeanSilveiraJS/ews-ept/blob/main/modelagem_evasao.ipynb)
e use *Ambiente de execução → Executar tudo*. O notebook instala o CatBoost, baixa os dados (cerca de 64 MB), confere o
SHA-256 e, ao final, oferece `resultados.zip` para download. Leva cerca de 10 minutos com a CPU padrão. Para usar o
novo modelo no protótipo, extraia o zip na raiz do projeto, substituindo `resultados/`.

**Localmente**, a partir da raiz do projeto:

```powershell
pip install -r requirements.txt
jupyter notebook modelagem_evasao.ipynb
```

As saídas são gravadas em `resultados/`, onde o protótipo já as encontra.

## Executar o protótipo

```powershell
cd prototipo
pip install -r requirements.txt
streamlit run app.py
```

Testes: `python -X utf8 test_app.py` dentro de `prototipo/`.

Páginas: *Painel da unidade* (para a equipe da unidade) e, em *Detalhes técnicos*, *Extrair dos microdados* (requer os
microdados em `dados/`) e *Como o modelo decide*.
Detalhes em [prototipo/README.md](prototipo/README.md).

## Principais resultados (teste: 21.315 matrículas de 618 turmas)

| Medida | Valor |
|---|---|
| ROC-AUC / precisão média (AP) | 0,767 / 0,710 |
| ROC-AUC dentro da unidade de ensino (medida principal) | 0,641 |
| ROC-AUC dentro da unidade, só cadastros completos | 0,700 |
| Regra simples (tipo de oferta) / regressão logística, dentro da unidade | 0,555 / 0,585 |

## Análises fora do notebook

Duas análises exploratórias, citadas no artigo, não são reproduzidas pelo notebook (os valores podem diferir em até
±0,005 dos do notebook):

- **Com e sem os cinco campos do cadastro** (tabela de variantes do artigo): com eles, o ROC-AUC dentro da unidade
  sobe de 0,644 para 0,825, mas, nas matrículas com cadastro completo, fica em 0,697 contra 0,689 sem eles. O ganho
  vinha do campo vazio, não do estudante.
- **Divisão aleatória por matrícula** em vez de por turma: o ROC-AUC sobe de 0,766 para 0,790, mas só a taxa de
  evasão da própria turma já chega a 0,769 nessa divisão, o que mostra que o ganho é aparente.

## Cuidados de interpretação

- O escore ordena as matrículas; não é uma probabilidade calibrada. Os níveis de alerta (0,30 e 0,70) são
  experimentais.
- A avaliação é retrospectiva, com desfechos já ocorridos.
- Dentro de uma unidade, a capacidade de ordenar estudantes é modesta (0,64) e, dentro de uma turma, fica pouco acima
  do acaso. O escore é ponto de partida para o acompanhamento, não base para decisões individuais.
- Forma de ingresso, identidade de gênero, tipo de NEE, renda e cor/raça ficaram fora do modelo: estão vazios com
  muito mais frequência entre os evadidos, provavelmente porque o cadastro foi completado durante o curso.
- SHAP descreve como o modelo chegou ao escore, não causas da evasão.
