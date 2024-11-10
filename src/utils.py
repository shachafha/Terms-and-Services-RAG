import json
import os
import time
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
    """Load API keys for Cohere, Pinecone, and Gemini services from a JSON file.

    :return: Tuple containing Cohere, Pinecone, and Gemini API keys as strings.
    """
    with open("api_keys.json") as f:
        api_keys = json.load(f)
    return api_keys["cohere"], api_keys["pinecone"], api_keys["gemini"]


def load_index_configurations():
    """Load index configurations from a JSON file.

    :return: Dictionary containing index configurations.
    """
    with open("index_configure.json") as f:
        return json.load(f)


def load_available_companies(data_directory='dataset'):
    """Load the list of available companies from a specified directory.

    :param data_directory: Path to the directory containing company subdirectories.
    :return: List of company names.
    """
    return [company for company in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, company))]


def initialize_pinecone(api_key):
    """Initialize and return a Pinecone client using the provided API key.

    :param api_key: API key for Pinecone.
    :return: Pinecone client instance.
    """
    return Pinecone(api_key=api_key)


@st.cache_resource
def load_models():
    """Load SentenceTransformer and HuggingFace models for generating embeddings and responses.

    :return: Dictionary of sentence transformer models and dictionary of HuggingFace models.
    """
    sentence_transformer_models = {
        "all-MiniLM-L6-v2": SentenceTransformer('all-MiniLM-L6-v2')
    }

    hf_models = {
        "Qwen2.5-0.5B-Instruct": {
            "model": AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device_map="auto",
                                                          torch_dtype=torch.float16),
            "tokenizer": AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        },
        "Gemini-1.5-flash": {
            "model": genai.GenerativeModel(model_name='gemini-1.5-flash'),
            "tokenizer": None
        }
    }
    return sentence_transformer_models, hf_models


def query_index(index, query_embedding, selected_company, top_k=5):
    """Query a Pinecone index to retrieve top K results based on the query embedding.

    :param index: Pinecone index to query.
    :param query_embedding: Query embedding vector.
    :param selected_company: Company name to filter the results.
    :param top_k: Number of top results to retrieve.
    :return: Query results from the index.
    """
    return index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter={"company_name": {"$eq": selected_company}}
    )


def generate_answer(rag_model, query, context, cohere_api_key, hf_models, rag_flag):
    """Generate an answer using specified RAG model with context and query.

    :param rag_model: The RAG model name (e.g., "Cohere", "Qwen2.5-0.5B-Instruct", "Gemini-1.5-flash").
    :param query: User's question to be answered.
    :param context: Context information from the company's T&C document.
    :param cohere_api_key: API key for Cohere.
    :param hf_models: Dictionary containing HuggingFace models.
    :param rag_flag: Boolean flag indicating whether to use context in the response generation.
    :return: Generated response as a string.
    """
    prompt = (f"You are an AI assistant created to assist users by answering inquiries regarding the Terms and "
              f"Conditions of various companies based on your knowledge. The question is: {query}. Please provide a response.")
    rag_prompt = (f"You are an AI assistant created to assist users by answering inquiries regarding the Terms and "
                  f"Conditions of various companies based on your knowledge. The question is: {query}. "
                  f"Additional Context from the company's T&C document: {context}. Please provide a response based on the context given.")

    # Generate response using selected RAG model
    if rag_model == "Cohere (command-r-plus)":
        # limited to 5 requests per minute
        time.sleep(0.5)
        response = cohere.Client(cohere_api_key).generate(
            model="command-r-plus",
            prompt=rag_prompt if rag_flag else prompt
        )
        return response.generations[0].text.strip()

    elif rag_model == "Qwen2.5-0.5B-Instruct":
        # Processing using Qwen model
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


def rerank_documents(query, docs, top_n=3, model_name='paraphrase-multilingual-mpnet-base-v2'):
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
    return [docs[i] for i in top_indices]


def rewrite_query(query, hf_models):
    """
    Rewrite the user query for improved retrieval performance in a RAG system.

    :param query: The original user query.
    :param hf_models: Dictionary containing the models.
    :return: Reformulated query as a string.
    """
    model = hf_models["Gemini-1.5-flash"]["model"]
    response = model.generate_content(f'You are an AI assistant tasked with reformulating a user query to improve '
                                      f'retrieval in a RAG system. Given the original query, rewrite it to be more '
                                      f'specific and likely to retrieve relevant information. Original query: {query}.'
                                      f'Provide only the reformulated query.')
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
    time.sleep(1)
    return response.text

def embed_query(query, embedding_model):
    """
    Generate embedding for a query using a specified embedding model.

    :param query: The user query.
    :param embedding_model: The embedding model to use.
    :return: Embedding vector as a list.
    """
    return embedding_model.encode([query], convert_to_tensor=True).tolist()[0]


def check_excel_valid(df, load_available_companies):
    """
    Validate the Excel file's format and contents, ensuring required columns and valid company names.

    :param df: DataFrame of the uploaded Excel file.
    :param load_available_companies: Function to load available companies.
    :return: Boolean indicating the file's validity.
    """
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


def retrieve_context(pc, index_name, query_embedding, selected_company, top_k, use_reranking, query):
    """
    Retrieve and optionally rerank context from a specified Pinecone index for a given query embedding.

    :param pc: Pinecone client instance.
    :param index_name: Name of the Pinecone index.
    :param query_embedding: Query embedding vector.
    :param selected_company: Company name for filtering results.
    :param top_k: Number of top results to retrieve.
    :param use_reranking: Boolean flag to rerank results.
    :param query: Original query string.
    :return: Tuple containing numbered context and similarity scores.
    """
    seperator = "-" * 50
    index = pc.Index(index_name)
    results = query_index(index, query_embedding, selected_company, top_k=top_k)
    # Reranking or showing top results
    if use_reranking:
        reranked_results = rerank_documents(query, results["matches"], top_n=3)
        numbered_context = "\n".join(
            [f"{i + 1}. {item['metadata']['text']}\n{seperator}" for i, item in enumerate(reranked_results)])
        similarity_scores = [item['score'] for item in reranked_results]
    else:
        numbered_context = "\n ".join(
            [f"{i + 1}. {item['metadata']['text']}\n{seperator}" for i, item in enumerate(results["matches"])])
        similarity_scores = [item['score'] for item in results['matches']]
    return numbered_context, similarity_scores


def collect_rag_responses(pc, index_names, query_embedding, selected_company, top_k, use_reranking, query,
                          cohere_api_key, hf_models, rag_model="Gemini-1.5-flash"):
    """
    Collect responses from RAG models across multiple indexes and generate answers for each.

    :param pc: Pinecone client instance.
    :param index_names: List of index names to query.
    :param query_embedding: Embedding vector for the query.
    :param selected_company: Company name to filter results.
    :param top_k: Number of top results to retrieve from each index.
    :param use_reranking: Boolean indicating if reranking should be applied.
    :param query: Original user query.
    :param cohere_api_key: API key for Cohere.
    :param hf_models: Dictionary containing HuggingFace models.
    :param rag_model: RAG model name to use for answer generation.
    :return: Tuple containing dictionary of RAG answers, list of contexts, and similarity scores.
    """
    rag_answers = {}
    rag_context = []
    rag_similarity_scores = []
    for name in index_names:
        numbered_context, similarity_scores = retrieve_context(pc, name, query_embedding, selected_company, top_k,
                                                               use_reranking, query)
        # Generate RAG and direct answers
        rag_answer = generate_answer(rag_model, query, numbered_context, cohere_api_key, hf_models, True)
        rag_answers[name] = rag_answer
        rag_context.append(numbered_context)
        rag_similarity_scores.append(similarity_scores)
    return rag_answers, rag_context, rag_similarity_scores


def optimize_response(query, hf_models, rag_answers):
    """
    Optimize and select the most relevant answer from multiple RAG model responses.

    :param query: The original user query.
    :param hf_models: Dictionary containing HuggingFace models.
    :param rag_answers: Dictionary of answers from different RAG models.
    :return: Tuple containing the selected option number and the optimized answer.
    """
    seperator = "-" * 50
    labeled_answers = [f"Option {idx + 1}:\n\n {answer}" for idx, answer in enumerate(rag_answers.values())]
    answer_list = f"\n{seperator}\n".join(labeled_answers)
    prompt = (
        f"Question: {query}\n\n"
        "Below are answers from different RAG models that use Terms and Services context. Each answer may vary in relevance, "
        "accuracy, and specificity. Please carefully evaluate each answer independently based on how well it addresses the question. "
        "Assign a score from 1 to 10 for each answer, focusing on relevance, confidence, legal accuracy, and detail.\n\n"
        "After scoring, identify the answer with the highest score and respond ONLY with this exact format, strictly as follows:\n\n"
        "[Option]: the option number \n"
        "[RAG Answer]: the exact text of the answer that received the highest score\n\n"
        f"Options:\n\n{seperator}\n{answer_list}\n\n"
    )
    model = hf_models["Gemini-1.5-flash"]["model"]
    response = model.generate_content(prompt).text
    parts = response.split("[RAG Answer]:")
    option_part = int(parts[0].split("Option")[-1].strip())
    rag_answer_part = parts[1].strip()
    return option_part, rag_answer_part
