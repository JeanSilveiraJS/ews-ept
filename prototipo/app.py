"""Protótipo de Early Warning System (EWS) para evasão em cursos técnicos da Rede Federal.

Uso (na pasta prototipo/):
    pip install -r requirements.txt
    streamlit run app.py

Páginas:
- Painel da unidade (app_pages/painel.py): para a equipe de assistência estudantil. Lista os estudantes de uma
  unidade ordenados pelo escore do modelo, com nível de alerta, fator principal e a ficha de cada um.
- Extrair dos microdados e Como o modelo decide (app_pages/extrair.py e modelo.py): detalhes técnicos, com recortes
  reais da PNP e os números e limites do modelo.

A lógica fica em nucleo.py (modelo, preparo dos dados, explicações) e extracao.py (leitura dos microdados); o
modelo e as bases são lidos de ../resultados/prototipo/, gerados pelo notebook ../modelagem_evasao.ipynb.
"""
import streamlit as st

st.set_page_config(page_title='EWS | Alerta de evasão', page_icon=':material/monitoring:', layout='wide')
pagina = st.navigation({
    'Equipe da unidade': [
        st.Page('app_pages/painel.py', title='Painel da unidade', icon=':material/groups:', default=True,
                help='Estudantes de uma unidade ordenados pelo escore, com nível de alerta, fator principal e ficha '
                     'individual.'),
    ],
    'Detalhes técnicos': [
        st.Page('app_pages/extrair.py', title='Extrair dos microdados', icon=':material/dataset:',
                help='Recorte real da PNP por instituição, unidade e turma, com comparação entre níveis e desfechos.'),
        st.Page('app_pages/modelo.py', title='Como o modelo decide', icon=':material/model_training:',
                help='Atributos, desempenho no teste, campos excluídos do cadastro e decomposição de um escore.'),
    ],
}, position='top')
pagina.run()
