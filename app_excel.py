import streamlit as st
import pandas as pd
from utils import *
import time
import os


cohere_api_key, pinecone_api_key, gemini_api_key = load_api_keys()
index_configurations = load_index_configurations()
sentence_transformer_models, hf_models = load_models()

# Initialize Pinecone
pc = initialize_pinecone(pinecone_api_key)
# Initialize Gemini
genai.configure(api_key=gemini_api_key)

# Sidebar options
with st.sidebar:
    use_optimizer = st.checkbox("Use Optimizer Strategy", help="Enable this option to try out all indexes and "
                                                               "have an LLM evaluate all RAG answers and choose the best one")
    index_names = {config['display_name']: config['index_name'] for config in index_configurations}
    selected_index_name = st.selectbox("Select Index", index_names.keys(), disabled=use_optimizer)
    selected_index_config = index_names[selected_index_name]
    embedding_model = sentence_transformer_models['all-MiniLM-L6-v2']
    rag_models = ["Gemini-1.5-flash", "Qwen2.5-0.5B-Instruct", "Cohere (command-r-plus)"]
    selected_rag_model = st.selectbox("Select RAG Model", rag_models, disabled=use_optimizer)
    use_rewrite = st.toggle("Rephrase Query", value=False, key='rewrite_checkbox',
                            help="Enable this option to allow the model to rephrase your query.")
    use_enrich = st.toggle("Enrich Query", value=False, key='enrich_checkbox',
                           help="Enable this to automatically include important keywords in your query for better search results.")
    use_reranking = st.toggle("Rerank Chunks", value=False, key='rerank_checkbox',
                              help="Enable this option to re-evaluate and rerank them based on their similarity to your question.")

# Info icon/message
st.info("**Instructions for Uploading Excel File:**\n"
        "1. The uploaded file must be in Excel format (.xlsx).\n"
        "2. Ensure the following columns are present:\n"
        "   - **company**: Name of the company.\n"
        "   - **question**: The question related to the Terms & Conditions.\n"
        "   - **right answer**: The correct answer to the question.\n"
        "3. Please ensure that there are no empty rows in the file.")

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
if st.button("Submit") and uploaded_file:
    with st.spinner("Searching through the Terms and Conditions..."):

        df = pd.read_excel(uploaded_file)
        st.write("Preview of the uploaded file:", df.head())

        if not check_excel_valid(df, load_available_companies):
            st.stop()

        else:
            responses = []
            batch_size = 5  # Number of rows to process at a time
            for idx, row in df[:1].iterrows():
                query = row['question']
                company = row['company']
                right_answer = row['right answer']
                original_query = query
                # Process query based on the flags for rewriting and enrichment
                if use_rewrite:
                    query = rewrite_query(query, hf_models)

                if use_enrich:
                    enriched = enrich_query(query, hf_models)
                query_embedding = embed_query(enriched if use_enrich else query, embedding_model)
                top_k = 5 if use_reranking else 3
                if use_optimizer:
                    selected_rag_model = "Gemini-1.5-flash"
                    rag_answers, rag_context, rag_similarity_scores = collect_rag_responses(pc,
                                                                                            index_names.values(),
                                                                                            query_embedding,
                                                                                            company, top_k,
                                                                                            use_reranking, query,
                                                                                            cohere_api_key,
                                                                                            hf_models)
                    # Get the optimal RAG answer
                    option_num, rag_answer = optimize_response(query, hf_models, rag_answers)
                    numbered_context = rag_context[option_num - 1]
                    similarity_score = rag_similarity_scores[option_num - 1]
                    time.sleep(20)
                else:
                    numbered_context, similarity_score = retrieve_context(pc, selected_index_config,
                                                                          query_embedding, company, top_k,
                                                                          use_reranking, query)
                    rag_answer = generate_answer(selected_rag_model, query, numbered_context, cohere_api_key,
                                                 hf_models,
                                                 True)
                direct_answer = generate_answer(selected_rag_model, query, "", cohere_api_key, hf_models, False)
                response = {
                    'Question': original_query, 'Company': company, 'Context': numbered_context,
                    'RAG Answer': rag_answer, "Direct Answer": direct_answer, "Right Answer": right_answer,
                    "Similarity Score": similarity_score
                }
                if use_rewrite:
                    response['Rephrased Question'] = query
                if use_optimizer:
                    response['Optimal Index'] = index_configurations[option_num - 1]['display_name']
                responses.append(response)

                # Pause every `batch_size` rows
                if (idx + 1) % batch_size == 0:
                    st.write('sleeping')
                    time.sleep(30)

            # Creating and saving the result dataframe
            result_df = pd.DataFrame(responses)
            st.write("Preview of the results:", result_df.head())

            output_dir = "model_responses"
            # Remove trailing underscores from the generated file name
            output_file = f"{selected_index_config if not use_optimizer else 'optimizer'}_{selected_rag_model}_{'rerank' if use_reranking else ''}_{'rephrase' if use_rewrite else ''}_{'enrich' if use_enrich else ''}.xlsx".replace(
                "__", "_").strip("_")
            output_path = os.path.join(output_dir, output_file)
            result_df.to_excel(output_path, index=False)

        st.balloons()

        st.download_button(
            label="Download the results file",
            data=open(output_path, 'rb').read(),
            file_name=output_file,
            mime="application/vnd.ms-excel"
        )
