import json
import os
import time
import zipfile
import torch
from pinecone import Pinecone
import cohere
from transformers import pipeline,AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import streamlit as st

def load_api_keys():
    with open("cohere_api_key.txt") as f:
        cohere_api_key = f.read().strip()

    with open("pinecone_api_key.txt") as f:
        pinecone_api_key = f.read().strip()

    return cohere_api_key, pinecone_api_key


def load_index_configurations():
    with open("index_configure.json") as f:
        return json.load(f)


def load_available_companies(data_directory='only_english_data'):
    return [company for company in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, company))]


def initialize_pinecone(api_key):
    return Pinecone(api_key=api_key)


def load_models():
    sentence_transformer_models = {
        "all-MiniLM-L6-v2": SentenceTransformer('all-MiniLM-L6-v2'),
        "paraphrase-MiniLM-L6-v2": SentenceTransformer('paraphrase-MiniLM-L6-v2')
    }

    hf_models = {
        "gpt2": pipeline('text-generation', model='gpt2', device=0),
        "Qwen2.5-0.5B-Instruct": {
            "model": AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                                          device_map="auto",torch_dtype=torch.float16),
            "tokenizer": AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        }
    }

    return sentence_transformer_models, hf_models


def query_index(index, query_embedding, selected_company, top_k=5):
    return index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter={"company_name": {"$eq": selected_company}}
    )


def generate_answer(rag_model, query, context, cohere_api_key, hf_models):
    if rag_model == "Cohere (command-r-plus)":
        # limited to 5 requests per minute
        time.sleep(0.5)
        response = cohere.Client(cohere_api_key).generate(
            model="command-r-plus",
            prompt=f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        )
        return response.generations[0].text.strip()
    elif rag_model == "GPT-2":
        generator = hf_models["gpt2"]
        response = generator(f"Context: {context}\n\nQuestion: {query}\nAnswer:",max_new_tokens=200,
                             num_return_sequences=1)
        return response[0]['generated_text'].split("Answer:")[1].strip()
    elif rag_model == "Qwen2.5-0.5B-Instruct":
        model = hf_models["Qwen2.5-0.5B-Instruct"]["model"]
        tokenizer = hf_models["Qwen2.5-0.5B-Instruct"]["tokenizer"]

        # Format the prompt and context using the chat template
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate response
        generated_ids = model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response


def zip_company_folder(company_name):
    zip_file_path = f"{company_name}.zip"
    company_folder_path = os.path.join('only_english_data', company_name)

    # Create a zip file of the company's folder
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(company_folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, company_folder_path)
                zipf.write(full_path, relative_path)

    return zip_file_path


def rerank_documents(query: str, docs, top_n: int = 3,
                     model_name: str = 'paraphrase-multilingual-mpnet-base-v2') :
    """
    Rerank documents using a different embedding model and similarity metric.

    :param query: The query string
    :param docs: List of documents  to rerank
    :param top_n: Number of top documents to return
    :param model_name: Name of the sentence transformer model to use
    :return: List of top_n most relevant Document objects
    """
    # Load the sentence transformer model
    model = SentenceTransformer(model_name)

    # Encode the query and documents
    query_embedding = model.encode(query)
    doc_embeddings = model.encode([doc["metadata"]["text"] for doc in docs])

    # Compute Euclidean distances
    distances = euclidean_distances([query_embedding], doc_embeddings)[0]

    # Convert distances to similarities (smaller distance = higher similarity)
    similarities = 1 / (1 + distances)

    # Sort the results in order of similarity
    top_indices = np.argsort(similarities)[::-1][:top_n]

    # Return the top N documents
    return [docs[i] for i in top_indices]


def rewrite_query(query, cohere_api_key):
    """
    Uses Cohere to rewrite a query for better retrieval in a RAG system.

    Parameters:
    - query (str): The original user query.
    - cohere_api_key (str): API key for accessing Cohere's service.

    Returns:
    - str: The rewritten query.
    """
    query_rewrite_template = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. 
    Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.

    Original query: {original_query}

    Rewritten query:"""

    # Create the prompt by inserting the original query into the template
    prompt = query_rewrite_template.format(original_query=query)

    # Initialize the Cohere client
    cohere_client = cohere.Client(cohere_api_key)

    # limited to 5 requests per minute
    time.sleep(0.5)

    # Generate the rewritten query using Cohere's model
    response = cohere_client.generate(
        model="command-r-plus",
        prompt=prompt
    )
    # Extract and return the rewritten query from the response
    return response.generations[0].text.strip()

def check_excel_valid(df,load_available_companies):
    if not 'company' in df.columns:
        st.error("The column 'company' is missing in the file.")
        return False

    if not 'question' in df.columns:
        st.error("The column 'question' is missing in the file.")
        return False

    if not 'right answer' in df.columns:
        st.error("The column 'right answer' is missing in the file.")
        return False

    # Find companies in the df that are not in available companies
    invalid_companies = df[~df['company'].isin(load_available_companies())]

    # If there are any invalid companies, display an error message
    if not invalid_companies.empty:
        invalid_list = invalid_companies['company'].tolist()
        st.error(f"The following companies are not in the list: {', '.join(invalid_list)}")
        return False
    return True
