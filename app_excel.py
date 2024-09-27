import streamlit as st
import pandas as pd
import time
from utils import load_api_keys, load_index_configurations, initialize_pinecone, load_models, query_index, generate_answer,rerank_documents,load_available_companies

def main():
    cohere_api_key, pinecone_api_key = load_api_keys()
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
        rag_models = {"Cohere (command-r-plus)": "command-r-plus", "GPT-2": "gpt2",
                      "Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B-Instruct"}
        selected_rag_model = st.selectbox("Select RAG Model", rag_models.keys())
        # Checkbox for reranking
        use_reranking = st.checkbox("Use reranking", value=False)

    # File upload and processing
    uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("Preview of the uploaded file:", df.head())

        # Find companies in the df that are not in available companies
        invalid_companies = df[~df['company'].isin(load_available_companies())]

        # If there are any invalid companies, display an error message
        if not invalid_companies.empty:
            invalid_list = invalid_companies['company'].tolist()
            st.error(f"The following companies are not in the list: {', '.join(invalid_list)}")

        else:
            responses = []
            for index, row in df.iterrows():
                query = row['question']
                company = row['company']
                query_embedding = embedding_model.encode([query], convert_to_tensor=True).tolist()[0]
                index = pc.Index(selected_index_name)
                top_k = 3 if selected_index_name == "semantic-200-index" else 5
                results = query_index(index, query_embedding, company, top_k=top_k)
                if use_reranking:
                    reranked_results = rerank_documents(query, results["matches"], top_n=top_k - 1)
                    context = "\n".join([result["metadata"]["text"] for result in reranked_results])
                    numbered_context = "\n".join(
                        [f"{i + 1}. {item['metadata']['text']}" for i, item in enumerate(reranked_results)])
                else:
                    context = "\n".join([result["metadata"]["text"] for result in results["matches"]])
                    numbered_context = "\n".join(
                        [f"{i + 1}. {item['metadata']['text']}" for i, item in enumerate(results["matches"])])

                rag_answer = generate_answer(selected_rag_model, query, context, cohere_api_key, hf_models)
                responses.append({'Question': query, 'Company': company, 'Context': numbered_context, 'RAG Answer': rag_answer})
                time.sleep(0.5)

            result_df = pd.DataFrame(responses)
            output_file = 'responses.xlsx'
            result_df.to_excel(output_file, index=False)

            st.write(f"Download the responses file:")
            st.download_button(label="Download Excel file", data=open(output_file, 'rb').read(), file_name=output_file, mime="application/vnd.ms-excel")
