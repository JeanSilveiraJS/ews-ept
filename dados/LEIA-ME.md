# Dados

Microdados de Eficiência Acadêmica da Plataforma Nilo Peçanha (PNP). São dados públicos, por matrícula, sem nome
ou CPF. Ainda assim, contêm códigos de matrícula e atributos sensíveis: não redistribua recortes individualizados.

| Arquivo | Uso | Origem | SHA-256 |
|---|---|---|---|
| `microdados_eficiencia_academica_2025.csv.gz` | Notebook (toda a modelagem). Ano-base 2025: ciclos com término previsto até 2024; 1.290.968 linhas e 79 colunas | https://d236w85zd3t8iw.cloudfront.net/pnp-tests/microdados/2025/microdados_eficiencia_academica_2025.csv.gz (acesso em 17/11/2026) | `18e9ad28b486c4793ce41f92e7b113c8a931aff522673707614db35687f00bed` |
| `microdados_eficiencia_academica_2024.csv.gz` | Apenas a página de extração do protótipo (demonstração com outra edição) | https://d236w85zd3t8iw.cloudfront.net/pnp-tests/microdados/2024/microdados_eficiencia_academica_2024.csv.gz (acesso em 18/11/2026) | `f8bdd8aa71b6be2cd05e00e3a991a119a381d18b210650cfa8d87a7c440763cd` |

- **Ano-base × edição:** o arquivo rotulado 2025 corresponde ao ano-base 2025 (coluna `Ano` = 2025).
- **Procedência:** os arquivos vêm do diretório de distribuição do portal de dados abertos da PNP:
  `https://app.powerbi.com/view?r=eyJrIjoiZDhkNGNiYzgtMjQ0My00OGVlLWJjNzYtZWQwYjI2OThhYWM1IiwidCI6IjllNjgyMzU5LWQxMjgtNGVkYi1iYjU4LTgyYjJhMTUzNDBmZiJ9`
- **Os arquivos `.csv.gz` não estão no repositório** (cerca de 94 MB). O notebook baixa o de 2025 da URL acima, se ele
  não estiver nesta pasta, e confere o SHA-256. Para a página *Extrair dos microdados* do protótipo, baixe os dois
  arquivos para esta pasta, com os nomes da tabela:

  ```powershell
  curl.exe -L -o dados/microdados_eficiencia_academica_2025.csv.gz https://d236w85zd3t8iw.cloudfront.net/pnp-tests/microdados/2025/microdados_eficiencia_academica_2025.csv.gz
  curl.exe -L -o dados/microdados_eficiencia_academica_2024.csv.gz https://d236w85zd3t8iw.cloudfront.net/pnp-tests/microdados/2024/microdados_eficiencia_academica_2024.csv.gz
  ```
