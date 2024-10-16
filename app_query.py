import streamlit as st
from utils import *


def main():
    # Load API keys and models
    cohere_api_key, pinecone_api_key, gemini_api_key = load_api_keys()
    index_configurations = load_index_configurations()
    available_companies = load_available_companies()
    sentence_transformer_models, hf_models = load_models()

    # Initialize Pinecone
    pc = initialize_pinecone(pinecone_api_key)
    # Initialize Gemini
    genai.configure(api_key=gemini_api_key)

    # Sidebar for selecting options
    with st.sidebar:
        st.title("Options")

        # Select indexing configuration
        index_names = [config['index_name'] for config in index_configurations]
        selected_index_name = st.selectbox("Select Index", index_names)
        selected_index_config = next(
            config for config in index_configurations if config['index_name'] == selected_index_name)
        selected_embedding_model_name = selected_index_config['embedding_model']
        embedding_model = sentence_transformer_models[selected_embedding_model_name]

        # Let the user choose RAG model and company
        rag_models = {"Gemini-1.5-flash": "Gemini-1.5-flash",
                      "Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-Instruct",
                      "Cohere (command-r-plus)": "command-r-plus"}
        selected_rag_model = st.selectbox("Select RAG Model", rag_models.keys())
        selected_company = st.selectbox("Select Company", available_companies)

        # Checkbox for reranking
        use_reranking = st.checkbox("Use reranking", value=False)

        # Checkbox for rewriting the query
        use_rewrite = st.checkbox("Rephrase the query", value=False)

        # Checkbox for enriching the query with keywords
        use_enrich = st.checkbox("Enrich the query with keywords", value=False)

    # Main area for user query
    col1, col2 = st.columns([1, 10])
    with col1:
        st.image('logo.jpeg')
    with col2:
        st.title("Terms and Services Query Interface")

    query = st.text_input("Enter your query:")

    if st.button("Submit") and query:
        if use_rewrite and use_enrich:
            query = rewrite_query(query, hf_models)
            st.markdown("### Rephrased Query")
            st.write(query)
            enriched = enrich_query(query, hf_models)

        elif use_rewrite:
            query = rewrite_query(query, hf_models)
            st.markdown("### Rephrased Query")
            st.write(query)

        elif use_enrich:
            enriched = enrich_query(query, hf_models)

        query_embedding = embed_query(enriched if use_enrich else query, embedding_model)
        index = pc.Index(selected_index_name)
        top_k = 5 if use_reranking else 3
        results = query_index(index, query_embedding, selected_company, top_k=top_k)

        if use_reranking:
            reranked_results = rerank_documents(query, results["matches"], top_n=3)
            context = "\n".join([result["metadata"]["text"] for result in reranked_results])
            numbered_context = "\n".join(
                [f"{i + 1}. {item['metadata']['text']}\n" for i, item in enumerate(reranked_results)])
        else:
            context = "\n".join([result["metadata"]["text"] for result in results["matches"]])
            numbered_context = "\n".join(
                [f"{i + 1}. {item['metadata']['text']}\n" for i, item in enumerate(results["matches"])])

        rag_answer = generate_answer(selected_rag_model, query, context, cohere_api_key, hf_models, True)

        # Display answers and context
        st.markdown("### Rag answer")
        st.write(rag_answer)

        direct_answer = generate_answer(selected_rag_model, query, "", cohere_api_key, hf_models, False)
        st.markdown("### Direct answer")
        st.write(direct_answer)

        st.markdown("### Raw Context")
        with st.expander("Click to view raw context chunks"):
            st.write(numbered_context)

    zip_file_path = f"{selected_company}.zip"
    download_clicked = st.download_button(
        label=f"Download {selected_company}'s Raw Files",
        data=open(zip_company_folder(selected_company), 'rb').read(),
        file_name=zip_file_path,
        mime='application/zip'
    )

    if download_clicked:
        os.remove(zip_file_path)
