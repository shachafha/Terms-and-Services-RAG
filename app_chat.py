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

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    def response_generator(response):
        yield response


    with st.chat_message("assistant"):
        response = st.markdown("Hi there! 👋 \n\n Need help with terms, policies, or services? I'm here to assist you. "
                               "Please choose a company, select any optional features, and then type your question below. "
                               "Together, we’ll find the answers you need!")
        # Add assistant response to chat history

        # Options displayed as selectboxes and checkboxes
    rag_model = "Gemini-1.5-flash"
    embedding_model = sentence_transformer_models["all-MiniLM-L6-v2"]
    with st.sidebar:
        selected_company = st.selectbox("Select Company", available_companies, key='company_select')
        use_rewrite = st.toggle("Rephrase Query", value=False, key='rewrite_checkbox',
                                  help="Enable this option to allow the model to rephrase your query.")
        use_enrich = st.toggle("Enrich Query", value=False, key='enrich_checkbox',
                                 help="Enable this to automatically include important keywords in your query for better search results.")
        use_reranking = st.toggle("Rerank Chunks", value=False, key='rerank_checkbox',
                                    help="Enable this option to re-evaluate and rerank them based on their similarity to your question.")
        if st.button("Clear Chat"):
            st.session_state.messages = []

    # Accept user input
    query = st.chat_input("Enter your question...")

    # Create a container for the chat and selections
    with st.container():
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            if message.get('with_columns'):
                if message.get('which column') == 3:
                    col3, col4 = st.columns([2, 2])
                    with col3.chat_message(message["role"]):
                        col3.markdown("*RAG answer:*")
                        col3.markdown(message["content"])
                else:
                    with col4.chat_message(message["role"]):
                        col4.markdown("*Direct answer:*")
                        col4.markdown(message["content"])
            else:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        # Display assistant response in chat message container


        # When a query is entered
        if query:
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(query)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
            # Create a new chat message for the human (user)


            time.sleep(1)
            with st.chat_message("assistant"):
                response = st.write_stream(response_generator(
                    "Thank you for your question! I'm searching through the Terms and Conditions documents to find "
                "the most accurate answer for you. This may take a few moments as I review multiple sections to "
                "ensure a thorough and relevant response. Please hold on; I'll be with you shortly! 📁🔍📄"))
                 # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})


            with st.spinner("Searching through the Terms and Conditions..."):
                # # Handle rephrasing and enriching logic
                if use_rewrite:
                    query = rewrite_query(query, hf_models)
                    with st.chat_message("assistant"):
                        response = st.write_stream(response_generator(
                            f"I have rephrased your query to: \n\n{query}"))
                        # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

                if use_enrich:
                    enriched = enrich_query(query, hf_models)

                # Query embedding and retrieval
                query_embedding = embed_query(enriched if use_enrich else query, embedding_model)
                top_k = 5 if use_reranking else 3

                if use_reranking:
                    with st.chat_message("assistant"):
                        response = st.write_stream(response_generator("Reranking results..."))
                        # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})

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
                with col3.chat_message("assistant"):
                    col3.write_stream(response_generator("*RAG answer:*"))
                    response = col3.write_stream(response_generator(optimal_rag_answer))
                    # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response,'with_columns':True, 'which column':3})

                with col4.chat_message("assistant"):
                    col4.write_stream(response_generator("*Direct answer:*"))
                    response = col4.write_stream(response_generator(direct_answer))
                    # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response,'with_columns':True, 'which column':4})

                # # Optionally, display context
                # with st.expander("Click to view the RAGS raw context chunks"):
                #     st.write(rag_context[option_num - 1])
                # input_visible = False


