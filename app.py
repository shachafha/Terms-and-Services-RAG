import streamlit as st

# Set up the layout for navigation
st.set_page_config(page_title="Terms and Services Query Interface", layout="wide")
col1, col2 = st.columns([1, 10])
col1.image('logo.jpeg')
col2.title("SimplifAI T&S")
col2.subheader("Terms and Services Query Interface")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chat", "Testing Environment"])


if page == "Chat":
    from app_chat import main
    main()
else:
    from app_excel import main
    main()

