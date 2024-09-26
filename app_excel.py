import streamlit as st
import pandas as pd
import time
from utils import load_api_keys, load_index_configurations, load_available_companies, initialize_pinecone, load_models, query_index, generate_answer

def main():
    cohere_api_key, pinecone_api_key = load_api_keys()
    index_configurations = load_index_configurations()
    available_companies = load_available_companies()
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
        rag_models = {"Cohere (command-r-plus)": "command-r-plus", "GPT-2": "gpt2", "Hugging Face (minilm-uncased)": "minilm-uncased-squad2"}
        selected_rag_model = st.selectbox("Select RAG Model", rag_models.keys())

    # File upload and processing
    uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("Preview of the uploaded file:", df.head())

        responses = []
        for index, row in df.iterrows():
            question = row['question']
            company = row['company']
            query_embedding = embedding_model.encode([question], convert_to_tensor=True).tolist()[0]
            index = pc.Index(selected_index_name)
            results = query_index(index, query_embedding, company)
            context = "\n".join([result["metadata"]["text"] for result in results["matches"]])
            rag_answer = generate_answer(selected_rag_model, question, context, cohere_api_key, hf_models)
            responses.append({'Question': question, 'Company': company, 'Context': context, 'RAG Answer': rag_answer})
            time.sleep(0.5)

        result_df = pd.DataFrame(responses)
        output_file = 'responses.xlsx'
        result_df.to_excel(output_file, index=False)

        st.write(f"Download the responses file:")
        st.download_button(label="Download Excel file", data=open(output_file, 'rb').read(), file_name=output_file, mime="application/vnd.ms-excel")

if __name__ == "__main__":
    main()
