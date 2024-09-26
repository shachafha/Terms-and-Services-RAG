import streamlit as st
from utils import load_api_keys, load_index_configurations, load_available_companies, initialize_pinecone, load_models, query_index, generate_answer

def main():
    # Load API keys and models
    cohere_api_key, pinecone_api_key = load_api_keys()
    index_configurations = load_index_configurations()
    available_companies = load_available_companies()
    sentence_transformer_models, hf_models = load_models()

    # Initialize Pinecone
    pc = initialize_pinecone(pinecone_api_key)

    # Sidebar for selecting options
    with st.sidebar:
        st.title("Options")

        # Select indexing configuration
        index_names = [config['index_name'] for config in index_configurations]
        selected_index_name = st.selectbox("Select Index", index_names)
        selected_index_config = next(config for config in index_configurations if config['index_name'] == selected_index_name)
        selected_embedding_model_name = selected_index_config['embedding_model']
        embedding_model = sentence_transformer_models[selected_embedding_model_name]

        # Let the user choose RAG model and company
        rag_models = {"Cohere (command-r-plus)": "command-r-plus", "GPT-2": "gpt2", "Hugging Face (minilm-uncased)": "minilm-uncased-squad2"}
        selected_rag_model = st.selectbox("Select RAG Model", rag_models.keys())
        selected_company = st.selectbox("Select Company", available_companies)

    # Main area for user query
    st.title("Terms and Services Query Interface")
    query = st.text_input("Enter your query:")

    if st.button("Submit") and query:
        query_embedding = embedding_model.encode([query], convert_to_tensor=True).tolist()[0]
        index = pc.Index(selected_index_name)
        results = query_index(index, query_embedding, selected_company)
        context = "\n".join([result["metadata"]["text"] for result in results["matches"]])
        rag_answer = generate_answer(selected_rag_model, query, context, cohere_api_key, hf_models)

        # Display answers and context
        st.markdown("### Rag answer")
        st.write(rag_answer)

        if selected_rag_model in ["GPT-2", "Cohere (command-r-plus)"]:
            direct_answer = generate_answer(selected_rag_model, query, "", cohere_api_key, hf_models)
            st.markdown("### Direct answer")
            st.write(direct_answer)

        st.markdown("### Context")
        st.text_area("", value=context, height=300, max_chars=None)
