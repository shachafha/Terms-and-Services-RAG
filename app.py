import streamlit as st
import json
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import cohere

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
    }
    selected_rag_model = st.selectbox("Select RAG Model", rag_models.keys())

    # Let the user choose a company from the available list
    selected_company = st.selectbox("Select Company", available_companies)

# Main area for queries and answers
st.title("Terms and Services Query Interface")

# Input for user query
query = st.text_input("Enter your query:")

if query:
    # Generate query embedding using SentenceTransformer
    query_embedding = embedding_model.encode([query], convert_to_tensor=True).tolist()[0]
    st.write("Done encoding query.")
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
    st.write("Done querying index.")
    context = "\n".join([result["metadata"]["text"] for result in results["matches"]])

    # Placeholder for generating an answer using the selected RAG model
    def generate_answer(query, context=None):
        if selected_rag_model == "Cohere (command-r-plus)":
            response = cohere.Client(COHERE_API_KEY).generate(
                model="command-r-plus",
                prompt=f"Context: {context}\n\nQuestion: {query}\nAnswer:"
            )
            return response.generations[0].text.strip()
        else:
            return "Direct answer functionality with this model is not yet implemented."

    # Generate RAG and direct answers
    rag_answer = generate_answer(query, context)
    st.write("Done generating RAG answer.")
    direct_answer = generate_answer(query, None)
    st.write("Done generating direct answer.")

    # Display current query and answers
    st.markdown(f"### Query: {query}")
    st.markdown(f"**RAG Answer**: {rag_answer}")
    st.markdown(f"**Direct Answer**: {direct_answer}")
