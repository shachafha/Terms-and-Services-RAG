import time
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

    query = st.chat_input("Enter your question...")
    # Create a container for the chat and selections
    with st.container():
        # Start with assistant's greeting message
        assistant = st.chat_message("assistant")
        assistant.write(
            "Hi there! 👋 \n\n Need help with terms, policies, or services? I'm here to assist you. "
            "Please choose a company, select any optional features, and then type your question below. "
            "Together, we’ll find the answers you need!")

        # Options displayed as selectboxes and checkboxes
        rag_model = "Gemini-1.5-flash"
        embedding_model = sentence_transformer_models["all-MiniLM-L6-v2"]
        selected_company = st.selectbox("Select Company", available_companies, key='company_select')
        col1, col2, col3 = st.columns([2, 2, 8])
        use_rewrite = col1.toggle("Rephrase Query", value=False, key='rewrite_checkbox', help="Enable this option to allow the model to rephrase your query.")
        use_enrich = col2.toggle("Enrich Query", value=False, key='enrich_checkbox', help="Enable this to automatically include important keywords in your query for better search results.")
        use_reranking = col3.toggle("Rerank Chunks", value=False, key='rerank_checkbox', help="Enable this option to re-evaluate and rerank them based on their similarity to your question.")

        # When a query is entered
        if query:
            # Create a new chat message for the human (user)
            human = st.chat_message("human")
            human.write(query)

            time.sleep(1)
            assistant_searching = st.chat_message("assistant")
            assistant_searching.write(
                "Thank you for your question! I'm searching through the Terms and Conditions documents to find "
                "the most accurate answer for you. This may take a few moments as I review multiple sections to "
                "ensure a thorough and relevant response. Please hold on; I'll be with you shortly! 📁🔍📄")

            with st.spinner("Searching through the Terms and Conditions..."):
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
                top_k = 5 if use_reranking else 3

                if use_reranking:
                    assistant_rerank = st.chat_message("assistant")
                    assistant_rerank.write("Reranking results...")

                rag_answers = {}
                rag_context = []
                for name in index_names:
                    index = pc.Index(name)
                    results = query_index(index, query_embedding, selected_company, top_k=top_k)
                    # Reranking or showing top results
                    if use_reranking:
                        reranked_results = rerank_documents(query, results["matches"], top_n=3)
                        context = "\n".join([result["metadata"]["text"] for result in reranked_results])
                        numbered_context = "\n".join(
                            [f"{i + 1}. {item['metadata']['text']}\n" for i, item in enumerate(reranked_results)])
                    else:
                        context = "\n".join([result["metadata"]["text"] for result in results["matches"]])
                        numbered_context = "\n".join(
                            [f"{i + 1}. {item['metadata']['text']}\n" for i, item in enumerate(results["matches"])])

                    # Generate RAG and direct answers
                    rag_answer = generate_answer(rag_model, query, context, cohere_api_key, hf_models, True)
                    rag_answers[name] = rag_answer
                    rag_context.append(numbered_context)

                # Get the optimal RAG answer
                option_num, optimal_rag_answer = optimize_response(query, hf_models, rag_answers)
                # Get the direct answer
                direct_answer = generate_answer(rag_model, query, "", cohere_api_key, hf_models, False)

                # Display answers
                col3, col4 = st.columns([2, 2])
                with col3:
                    assistant_rag = st.chat_message("assistant")
                assistant_rag.markdown("*RAG answer:*")
                assistant_rag.write(optimal_rag_answer)
                with col4:
                    assistant_direct = st.chat_message("assistant")
                assistant_direct.write("*Direct answer:*")
                assistant_direct.write(direct_answer)

                # Optionally, display context
                with st.expander("Click to view the RAGS raw context chunks"):
                    st.write(rag_context[option_num - 1])
                input_visible = False

            st.chat_message("assistant").write(f"I hope my answer was helpful 😊! \n\n Feel free to ask another question. \
            Scroll back up to change the previous configurations.")
