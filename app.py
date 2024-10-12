import streamlit as st

# Set up the layout for navigation
st.set_page_config(page_title="Terms and Services Query Interface", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Query Interface", "Upload Excel"])

if page == "Query Interface":
    from app_query import main
    main()
elif page == "Upload Excel":
    from app_excel import main
    main()
