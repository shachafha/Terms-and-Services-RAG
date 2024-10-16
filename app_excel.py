import streamlit as st
import pandas as pd
from utils import *

def main():
    cohere_api_key, pinecone_api_key, gemini_api_key = load_api_keys()
    index_configurations = load_index_configurations()
    sentence_transformer_models, hf_models = load_models()
    pc = initialize_pinecone(pinecone_api_key)

    # Sidebar options
    with st.sidebar:
        st.title("Options")
        index_names = [config['index_name'] for config in index_configurations]
        selected_index_name = st.selectbox("Select Index", index_names)
        selected_index_config = next(config for config in index_configurations if config['index_name'] == selected_index_name)
        selected_embedding_model_name = selected_index_config['embedding_model']
        embedding_model = sentence_transformer_models[selected_embedding_model_name]
        rag_models = {"Gemini-1.5-flash": "Gemini-1.5-flash",
                      "Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-Instruct",
                      "Cohere (command-r-plus)": "command-r-plus"}
        selected_rag_model = st.selectbox("Select RAG Model", rag_models.keys())
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
        df = pd.read_excel(uploaded_file)
        st.write("Preview of the uploaded file:", df.head())

        if not check_excel_valid(df, load_available_companies):
            st.stop()

        else:
            responses = []
            for index, row in df.iterrows():
                query = row['question']
                company = row['company']
                right_answer = row['right answer']
                original_query = query
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
                results = query_index(index, query_embedding, company, top_k=top_k)
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
                direct_answer = generate_answer(selected_rag_model, query, "", cohere_api_key, hf_models, False)
                response = {'Question': original_query, 'Company': company, 'Context': numbered_context, 'RAG Answer': rag_answer, "Direct Answer": direct_answer, "Right Answer": right_answer}
                if use_rewrite:
                    response['Rephrased Question'] = query
                responses.append(response)

            result_df = pd.DataFrame(responses)
            if use_rewrite:
                result_df = result_df[['Question', 'Rephrased Question', 'Company', 'Context', 'RAG Answer', 'Direct Answer', 'Right Answer']]
            st.write("Preview of the results:", result_df.head())

            output_file = 'responses.xlsx'
            result_df.to_excel(output_file, index=False)

            st.download_button(label="Download the results file", data=open(output_file, 'rb').read(), file_name=output_file, mime="application/vnd.ms-excel")
