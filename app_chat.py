import streamlit as st
from utils import *


def main():
    # Load API keys and models
    cohere_api_key, pinecone_api_key, gemini_api_key = load_api_keys()
    index_configurations = load_index_configurations()
    available_companies = load_available_companies()
    sentence_transformer_models, hf_models = load_models()
    index_names = [config['index_name'] for config in index_configurations]
    rag_models = {"Gemini-1.5-flash": "Gemini-1.5-flash",
                  "Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-Instruct",
                  "Cohere (command-r-plus)": "command-r-plus"}

    # Initialize Pinecone
    pc = initialize_pinecone(pinecone_api_key)
    # Initialize Gemini
    genai.configure(api_key=gemini_api_key)

    # Main area for user query
    col1, col2 = st.columns([1, 10])
    with col1:
        st.image('logo.jpeg')
    with col2:
        st.title("Terms and Services Chat")

    input_visible = True
    # Input field for user query
    if input_visible:
        query = st.chat_input("Enter your question...")
        # Create a container for the chat and selections
        with st.container():
            # Start with assistant's greeting message
            assistant = st.chat_message("assistant")
            assistant.write(
                "Hello 👋 Welcome to the Terms and Services chat. \n\n Please select your options below and \
                ask any questions you have about terms, policies or services. I'm here to help you understand the details!")

            # Options displayed as selectboxes and checkboxes
            selected_index_name = st.selectbox("Select Index", index_names, key='index_select')
            selected_rag_model = st.selectbox("Select RAG Model", rag_models.keys(), key='model_select')
            selected_company = st.selectbox("Select Company", available_companies, key='company_select')
            use_reranking = st.checkbox("Use reranking", value=False, key='rerank_checkbox')
            use_rewrite = st.checkbox("Rephrase the query", value=False, key='rewrite_checkbox')
            use_enrich = st.checkbox("Enrich the query with keywords", value=False, key='enrich_checkbox')
            selected_index_config = next(
                config for config in index_configurations if config['index_name'] == selected_index_name)
            selected_embedding_model_name = selected_index_config['embedding_model']
            embedding_model = sentence_transformer_models[selected_embedding_model_name]
            # When a query is entered
            if query:
                # Create a new chat message for the human (user)
                human = st.chat_message("human")
                human.write(query)
                # Handle rephrasing and enriching logic
                if use_rewrite and use_enrich:
                    query = rewrite_query(query, hf_models)
                    assistant_rewrite = st.chat_message("assistant")
                    assistant_rewrite.write(f"I have rephrased your query to: \n\n{query}")
                    enriched = enrich_query(query, hf_models)

                elif use_rewrite:
                    query = rewrite_query(query, hf_models)
                    assistant_rewrite = st.chat_message("assistant")
                    assistant_rewrite.write(f"I have rephrased your query to: \n\n{query}")

                elif use_enrich:
                    enriched = enrich_query(query, hf_models)

                # Query embedding and retrieval
                query_embedding = embed_query(enriched if use_enrich else query, embedding_model)
                index = pc.Index(selected_index_name)
                top_k = 5 if use_reranking else 3
                results = query_index(index, query_embedding, selected_company, top_k=top_k)

                # Reranking or showing top results
                if use_reranking:
                    assistant_rerank = st.chat_message("assistant")
                    assistant_rerank.write("Reranking results...")
                    reranked_results = rerank_documents(query, results["matches"], top_n=3)
                    context = "\n".join([result["metadata"]["text"] for result in reranked_results])
                    numbered_context = "\n".join(
                        [f"{i + 1}. {item['metadata']['text']}\n" for i, item in enumerate(reranked_results)])
                else:
                    context = "\n".join([result["metadata"]["text"] for result in results["matches"]])
                    numbered_context = "\n".join(
                        [f"{i + 1}. {item['metadata']['text']}\n" for i, item in enumerate(results["matches"])])

                # Generate RAG and direct answers
                rag_answer = generate_answer(selected_rag_model, query, context, cohere_api_key, hf_models, True)
                direct_answer = generate_answer(selected_rag_model, query, "", cohere_api_key, hf_models, False)

                # Display answers
                col3, col4 = st.columns([2, 2])
                with col3:
                    assistant_rag = st.chat_message("assistant")
                assistant_rag.markdown("*Rag answer:*")
                assistant_rag.write(rag_answer)
                with col4:
                    assistant_direct = st.chat_message("assistant")
                assistant_direct.write("*Direct answer:*")
                assistant_direct.write(direct_answer)

                # Optionally, display context
                with st.expander("Click to view the RAGS raw context chunks"):
                    st.write(numbered_context)
                input_visible = False
    if not input_visible:
        st.chat_message("assistant").write(f"I hope my answer was helpful 😊! \n\n Feel free to ask another question. \
        Scroll back up to change the previous configurations.")

