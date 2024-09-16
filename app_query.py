import streamlit as st
import json
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import cohere
from transformers import pipeline

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
    available_companies = [company for company in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, company))]

    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Load SentenceTransformer models
    sentence_transformer_models = {
        "all-MiniLM-L6-v2": SentenceTransformer('all-MiniLM-L6-v2'),
        "paraphrase-MiniLM-L6-v2": SentenceTransformer('paraphrase-MiniLM-L6-v2')
    }

    # Load Hugging Face models
    hf_models = {
        "deepset/roberta-base-squad2": pipeline('question-answering', model="deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2"),
        "deepset/minilm-uncased-squad2": pipeline('question-answering', model="deepset/minilm-uncased-squad2", tokenizer="deepset/minilm-uncased-squad2")
    }

    # Set up the layout
    # Left sidebar for selection options
    with st.sidebar:
        st.title("Options")

        # Select indexing configuration
        index_names = [config['index_name'] for config in index_configurations]
        selected_index_name = st.selectbox("Select Index", index_names)

        # Get selected index configuration
        selected_index_config = next(config for config in index_configurations if config['index_name'] == selected_index_name)
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

        # Let the user choose a company from the available list
        selected_company = st.selectbox("Select Company", available_companies)

    # Main area for queries and answers
    st.title("Terms and Services Query Interface")

    # Input for user query
    query = st.text_input("Enter your query:")

    # Submit button for the query
    if st.button("Submit Query") and query:
        # Generate query embedding using SentenceTransformer
        query_embedding = embedding_model.encode([query], convert_to_tensor=True).tolist()[0]
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

        results = query_index(index, query_embedding, selected_company)
        context = "\n".join([result["metadata"]["text"] for result in results["matches"]])

        # Display the context in a scrollable text area
        st.text_area("Context", value=context, height=200)

        # Placeholder for generating an answer using the selected RAG model
        def generate_answer(query, context=None):
            if selected_rag_model == "Cohere (command-r-plus)":
                response = cohere.Client(COHERE_API_KEY).generate(
                    model="command-r-plus",
                    prompt=f"Context: {context}\n\nQuestion: {query}\nAnswer:"
                )
                return response.generations[0].text.strip()
            elif selected_rag_model == "Hugging Face (roberta-base)":
                nlp = hf_models["deepset/roberta-base-squad2"]
                response = nlp({'question': query, 'context': context})
                return response['answer']
            elif selected_rag_model == "Hugging Face (minilm-uncased)":
                nlp = hf_models["deepset/minilm-uncased-squad2"]
                response = nlp({'question': query, 'context': context})
                return response['answer']

        # Generate RAG and direct answers
        rag_answer = generate_answer(query, context)
        if selected_rag_model == "Cohere (command-r-plus)":
            direct_answer = generate_answer(query, "")
            st.markdown(f"**Direct Answer**: {direct_answer}")

        # Display current query and answers
        st.markdown(f"### Query: {query}")
        st.markdown(f"**RAG Answer**: {rag_answer}")
