import json
import os
import time
import zipfile
import torch
from pinecone import Pinecone
import cohere
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import streamlit as st


def load_api_keys():
    with open("cohere_api_key.txt") as f:
        cohere_api_key = f.read().strip()

    with open("pinecone_api_key.txt") as f:
        pinecone_api_key = f.read().strip()

    with open("gemini_api_key.txt") as f:
        gemini_api_key = f.read
    return cohere_api_key, pinecone_api_key, gemini_api_key


def load_index_configurations():
    with open("index_configure.json") as f:
        return json.load(f)


def load_available_companies(data_directory='only_english_data'):
    return [company for company in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, company))]


def initialize_pinecone(api_key):
    return Pinecone(api_key=api_key)


@st.cache_resource
def load_models():
    sentence_transformer_models = {
        "all-MiniLM-L6-v2": SentenceTransformer('all-MiniLM-L6-v2'),
        "paraphrase-MiniLM-L6-v2": SentenceTransformer('paraphrase-MiniLM-L6-v2')
    }

    hf_models = {
        "Qwen2.5-0.5B-Instruct": {
            "model": AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                                          device_map="auto", torch_dtype=torch.float16),
            "tokenizer": AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        },
        "Gemini-1.5-flash": {
            "model": genai.GenerativeModel(model_name='gemini-1.5-flash'),
            "tokenizer": None
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


def generate_answer(rag_model, query, context, cohere_api_key, hf_models, rag_flag):
    prompt = (f"You are an AI assistant created to assist users by answering inquiries regarding the Terms and \
                Conditions of various companies based on your knowledge. The question is: {query}.\
                Please provide a response.")
    rag_prompt = (f"You are an AI assistant created to assist users by answering inquiries regarding the Terms and \
                    Conditions of various companies based on your knowledge. The question is: {query}.\
                    Additional Context from the company's T&C document: {context}. \
                    Please provide a response based on the context given.")
    if rag_model == "Cohere (command-r-plus)":
        # limited to 5 requests per minute
        time.sleep(0.5)
        response = cohere.Client(cohere_api_key).generate(
            model="command-r-plus",
            # prompt=f"Context: {context}\n\nQuestion: {query}\nAnswer:"
            prompt=rag_prompt if rag_flag else prompt
        )
        return response.generations[0].text.strip()
    elif rag_model == "Qwen2.5-0.5B-Instruct":
        model = hf_models["Qwen2.5-0.5B-Instruct"]["model"]
        tokenizer = hf_models["Qwen2.5-0.5B-Instruct"]["tokenizer"]

        # Format the prompt and context using the chat template
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant designed to answer questions about Terms and "
                                          "Conditions documents based on context."},
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
    elif rag_model == "Gemini-1.5-flash":
        # limited to 15 requests per minute
        time.sleep(1)
        model = hf_models["Gemini-1.5-flash"]["model"]
        response = model.generate_content(rag_prompt if rag_flag else prompt)
        return response.text


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
                     model_name: str = 'paraphrase-multilingual-mpnet-base-v2'):
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


def rewrite_query(query, hf_models):
    """
    Uses Gemini to rewrite a query for better retrieval in a RAG system.

    Parameters:
    - query (str): The original user query.
    - hf_models (dict): A dictionary containing the models.

    Returns:
    - str: The rewritten query.
    """
    model = hf_models["Gemini-1.5-flash"]["model"]
    response = model.generate_content(f'You are an AI assistant tasked with reformulating a user query to improve\
     retrieval in a RAG system. Given the original query, rewrite it to be more specific and likely to \
     retrieve relevant information. Original query: {query}. Provide only the reformulated query.')
    time.sleep(1)
    return response.text


def enrich_query(query, hf_models):
    """
    Uses Gemini to extract keywords and concatenate them to the end of the query.

    Parameters:
    - query (str): The user query.
    - hf_models (dict): A dictionary containing the models.

    Returns:
    - str: The enriched query.
    """
    prompt = (f"Given a user query, your task is to identify and extract the most relevant keywords that would \
    typically appear in the relevant sections in Terms and Conditions documents related to the subject of the query.\
      The output format should look as follows: '<the original query> : [list of keywords]'\
     User query: {query}.")
    model = hf_models["Gemini-1.5-flash"]["model"]
    response = model.generate_content(prompt)
    st.write(response.text)
    time.sleep(1)
    return response.text


def embed_query(query, embedding_model):
    return embedding_model.encode([query], convert_to_tensor=True).tolist()[0]


def check_excel_valid(df, load_available_companies):
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
