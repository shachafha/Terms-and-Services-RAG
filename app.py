import streamlit as st
st.set_page_config(page_title="Terms and Services Query Interface", layout="wide", page_icon="logo.png")
col1, col2 = st.columns([1, 10])
col1.image('logo.jpeg')
col2.title("SimplifAI T&S")
col2.subheader("Terms and Services Query Interface")
chat = st.Page("app_chat.py", title="Query", icon=":material/smart_toy:")
test = st.Page("app_excel.py", title="Testing Environment", icon=":material/query_stats:")

pg = st.navigation([chat, test])
pg.run()
