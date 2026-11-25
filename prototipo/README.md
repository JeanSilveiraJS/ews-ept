# Protótipo EWS — painel de alerta de evasão

Aplicação Streamlit local que aplica o modelo de evasão (CatBoost, 21 atributos conhecidos no ingresso) aos
estudantes de cursos técnicos de uma unidade da Rede Federal. Não há banco de dados, API nem autenticação; nada é
gravado.

## Como executar

```powershell
cd prototipo
pip install -r requirements.txt
streamlit run app.py
```

Requer Python ≥ 3.10 e Streamlit ≥ 1.67. Verificado com Python 3.14.10, Streamlit 1.67.0 e CatBoost 1.2.10.

## Páginas

| Página | Grupo no menu | O que mostra |
|---|---|---|
| **Painel da unidade** | Equipe da unidade | Escolhe-se a instituição e a unidade (ficam no endereço da página: `?inst=IFMT&uni=Campus+Cuiabá`). Indicadores por nível; lista ordenada pelo escore com nível, curso e fator principal; ficha do estudante (posição na unidade, frase e gráfico com o que mais pesou, dados usados pelo modelo e, num painel recolhível, o perfil completo com os 21 atributos, o efeito de cada um e a turma e as datas do ciclo, quando disponíveis); aba *Visão por curso* com os níveis por curso e os fatores mais frequentes no nível alto; download da lista |
| **Extrair dos microdados** | Detalhes técnicos | Recorte real da PNP (ano-base 2025 ou 2024) por instituição, unidade, curso, situação e turmas (filtros também pelo endereço: `extrair?ex_ano=2025&ex_inst=IFMT&ex_uni=Campus+Cuiabá`); uso de cada turma no treinamento; níveis × desfecho observado; ROC-AUC do recorte e dentro das turmas; abre o recorte no painel ou baixa o CSV |
| **Como o modelo decide** | Detalhes técnicos | Atributos por grupo; comparação com alternativas (sem modelo, regra simples, regressão logística); evasão observada por nível; por que cinco campos do cadastro ficam de fora; importância dos atributos; decomposição SHAP de um escore; limites |

**Momento de uso pensado:** o início de cada ciclo (entrada de uma turma). O modelo usa só dados do ingresso, então
o escore de um estudante não muda ao longo do curso, a não ser que os dados do ingresso sejam corrigidos.

## Formato do CSV (opção *Arquivo CSV* do painel)

- **Colunas:** as 21 do esquema (baixe o *modelo de CSV* no painel) e, opcionalmente, `Identificação` (matrícula ou
  nome), que só é exibida. Diferenças de maiúsculas, acentos, espaços ou `_` são aceitas e avisadas; colunas extras
  são ignoradas; colunas repetidas geram erro.
- **Separador e codificação:** `,`, `;`, tabulação ou `|`; UTF-8 (com ou sem BOM) ou Latin-1.
- **Idade:** `Idade no ingresso`, a idade observada no ingresso (no treino ela foi estimada, com incerteza de até
  cerca de dois anos). Valores fora de 10–90 geram aviso.
- **Duração do ciclo:** `Duracao_Ciclo_Dias` pode ser trocada por `Data de Inicio do Ciclo` e
  `Data de Fim Previsto do Ciclo`.
- **Turma e datas (opcionais):** `Código do Ciclo Matricula`, `Data de Inicio do Ciclo` e `Data de Fim Previsto do Ciclo`
  aparecem no perfil completo da ficha; não entram no modelo. O CSV baixado da extração já as inclui.
- **Campos vazios:** categóricos viram "não informado" e numéricos ficam ausentes.

## Arquivos

Código (nesta pasta):

| Arquivo | Conteúdo |
|---|---|
| `app.py` | Entrada: configuração e navegação entre as páginas |
| `app_pages/` | `painel.py`, `extrair.py` e `modelo.py` (interface de cada página) |
| `nucleo.py` | Lógica sem interface: caminhos dos artefatos, modelo, leitura e preparo do CSV, níveis, lista e explicações |
| `comum.py` | Elementos de interface compartilhados (modelo em cache, cálculo dos escores, cores) |
| `extracao.py` | Leitura dos microdados de `../dados/`, resumo das turmas e conversão para o formato do painel |
| `test_app.py` | 16 testes automáticos |
| `.streamlit/config.toml` | Tema e opções do servidor |

Modelo e bases: lidos diretamente de `../resultados/prototipo/`, onde o notebook `../modelagem_evasao.ipynb` os grava
(seção 10). Não há cópia nesta pasta; se o notebook for executado de novo, o protótipo passa a usar os novos arquivos
(ajuste então os valores `ESPERADO_*` de `test_app.py`).

| Arquivo em `../resultados/prototipo/` | Conteúdo |
|---|---|
| `modelo_evasao.cbm`, `esquema_modelo.json` | Modelo (262 árvores) e esquema (21 atributos, grupos, limiares 0,30/0,70) |
| `resultados_modelo.json` | Números da avaliação, mostrados em *Como o modelo decide* (cópia de `../resultados/resultados.json`) |
| `base_demonstracao.csv` | Base padrão do painel: 1.829 matrículas de teste de 15 unidades de IFCE, IFES, IFMT, IFPA e IFSUL (as 3 maiores unidades do instituto com mais matrículas de teste em cada região), sem desfecho nem identificadores; retiradas as combinações de unidade, curso, sexo e deficiência com menos de 5 pessoas (35 registros) |
| `particao_ciclos_2025.csv` | Turma → partição (Treino/Validação/Teste) da edição 2025 |

## Edição 2024 na extração

O arquivo de 2024 não tem o tipo de deficiência nem o programa associado. O programa associado é derivado do tipo de
oferta (igual ao valor real em todo o teste de 2025). O tipo de deficiência é suposto "Não possui deficiência" (valor
de 98,4% das matrículas técnicas): no teste de 2025, essa suposição muda o nível de 0,7% das matrículas, enquanto
deixar o campo vazio, valor que o modelo nunca viu, mudaria 34%. A suposição aparece na página e na ficha
("(suposto)"). A edição serve para mostrar o processamento de outra edição, não como validação.

## Testes

```powershell
cd prototipo
python -X utf8 test_app.py          # ou: python -m pytest test_app.py
```

Conferem o modelo contra o esquema (21 atributos, sem os cinco campos do cadastro); a base de demonstração (colunas,
ausência de rótulo, níveis, regra dos 5, aditividade SHAP); a importação de CSV (separadores, codificações, nomes
alternativos, identificação, vazios, erros); a lista da unidade (ordem, posição, campos vazios, fator principal,
seleção); a ficha (explicação, frase, ordem do gráfico); as três páginas com `AppTest`; e a extração (reprodução exata
do IFMT – Campus Cuiabá da base de demonstração, edição 2024 e envio ao painel). Os testes de extração são pulados se
os microdados não estiverem em `../dados/`.

## Limites

Avaliação retrospectiva; ROC-AUC de 0,767 no teste, mas modesta dentro da unidade (0,641) e pouco acima do acaso
dentro da turma; escore não calibrado e níveis experimentais; SHAP descreve o modelo, não causas; forma de ingresso,
identidade de gênero, tipo de NEE, renda e cor/raça ficaram fora por preenchimento desigual entre concluintes e
evadidos; sem avaliação formal de equidade; uso apenas como apoio à equipe.

## Observação de desenvolvimento

Nesta pasta (Google Drive), o Streamlit nem sempre percebe alterações nos arquivos; depois de editar, reinicie o
`streamlit run`.
