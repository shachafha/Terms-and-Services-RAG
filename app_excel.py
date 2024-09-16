import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import cohere
from transformers import AutoModelForQuestionAnswering, pipeline
import time
import json
import os

def main():
    # Load API keys
    with open("cohere_api_key.txt") as f:
        COHERE_API_KEY = f.read().strip()

    with open("pinecone_api_key.txt") as f:
        PINECONE_API_KEY = f.read().strip()

    # Load index configurations
    with open("index_configure.json") as f:
        index_configurations = json.load(f)

    # Get available companies from the directory structure
    data_directory = 'only_english_data'
    available_companies = [company for company in os.listdir(data_directory) if
                           os.path.isdir(os.path.join(data_directory, company))]

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Load SentenceTransformer models
    sentence_transformer_models = {
        "all-MiniLM-L6-v2": SentenceTransformer('all-MiniLM-L6-v2'),
        "paraphrase-MiniLM-L6-v2": SentenceTransformer('paraphrase-MiniLM-L6-v2')
    }

    # Load Hugging Face models
    hf_models = {
        "deepset/roberta-base-squad2": pipeline('question-answering', model="deepset/roberta-base-squad2",
                                                tokenizer="deepset/roberta-base-squad2"),
        "deepset/minilm-uncased-squad2": pipeline('question-answering', model="deepset/minilm-uncased-squad2",
                                                  tokenizer="deepset/minilm-uncased-squad2")
    }

    st.title("Batch Query Interface")

    # Sidebar for selecting options
    with st.sidebar:
        st.title("Options")

        # Select indexing configuration
        index_names = [config['index_name'] for config in index_configurations]
        selected_index_name = st.selectbox("Select Index", index_names)

        # Get selected index configuration
        selected_index_config = next(
            config for config in index_configurations if config['index_name'] == selected_index_name)
        selected_embedding_model_name = selected_index_config['embedding_model']

        # Automatically select the embedding model from the JSON configuration
        embedding_model = sentence_transformer_models[selected_embedding_model_name]

        # Let the user choose the RAG model for generating answers
        rag_models = {
            "Cohere (command-r-plus)": "command-r-plus",
            "Hugging Face (roberta-base)": "roberta-base-squad2",
            "Hugging Face (minilm-uncased)": "minilm-uncased-squad2"
        }
        selected_rag_model = st.selectbox("Select RAG Model", rag_models.keys())

    # File upload and processing
    uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")

    if uploaded_file:
        # Read the Excel file
        df = pd.read_excel(uploaded_file)

        # Display the dataframe to the user
        st.write("Preview of the uploaded file:", df.head())

        # Placeholder for responses
        responses = []

        # Process each row in the dataframe
        for index, row in df.iterrows():
            question = row['question']
            company = row['company']

            # Generate query embedding using SentenceTransformer
            query_embedding = embedding_model.encode([question], convert_to_tensor=True).tolist()[0]
            # Query the Pinecone index, filtering by company
            index = pc.Index(selected_index_name)


            # Function to query index with company filter
            def query_index(index, query_embedding, selected_company, top_k=5):
                """Query Pinecone index with a query embedding and company filter."""
                return index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True,
                    filter={"company_name": {"$eq": selected_company}}
                )


            results = query_index(index, query_embedding, company)
            context = "\n".join([result["metadata"]["text"] for result in results["matches"]])


            # Generate an answer using the selected RAG model
            def generate_answer(query, context=None):
                if selected_rag_model == "Cohere (command-r-plus)":
                    response = cohere.Client(COHERE_API_KEY).generate(
                        model="command-r-plus",
                        prompt=f"Context: {context}\n\nQuestion: {query}\nAnswer:"
                    )
                    time.sleep(0.5)
                    return response.generations[0].text.strip()
                elif selected_rag_model == "Hugging Face (roberta-base)":
                    nlp = hf_models["deepset/roberta-base-squad2"]
                    response = nlp({'question': query, 'context': context})
                    return response['answer']
                elif selected_rag_model == "Hugging Face (minilm-uncased)":
                    nlp = hf_models["deepset/minilm-uncased-squad2"]
                    response = nlp({'question': query, 'context': context})
                    return response['answer']


            rag_answer = generate_answer(question, context)

            # Append the result
            responses.append({
                'Question': question,
                'Company': company,
                'Context': context,
                'RAG Answer': rag_answer
            })



        # Create a DataFrame for the results
        result_df = pd.DataFrame(responses)

        # Save the results to an Excel file
        output_file = 'responses.xlsx'
        result_df.to_excel(output_file, index=False)

        # Provide a download link
        st.write(f"Download the responses file:")
        st.download_button(
            label="Download Excel file",
            data=open(output_file, 'rb').read(),
            file_name=output_file,
            mime="application/vnd.ms-excel"
        )
