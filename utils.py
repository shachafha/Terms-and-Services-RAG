import json
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import cohere
from transformers import pipeline


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
        "gpt2": pipeline('text-generation', model='gpt2'),
        "deepset/minilm-uncased-squad2": pipeline('question-answering', model="deepset/minilm-uncased-squad2",
                                                  tokenizer="deepset/minilm-uncased-squad2")
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
        response = cohere.Client(cohere_api_key).generate(
            model="command-r-plus",
            prompt=f"Context: {context}\n\nQuestion: {query}\nAnswer:"
        )
        return response.generations[0].text.strip()
    elif rag_model == "GPT-2":
        generator = hf_models["gpt2"]
        response = generator(f"Context: {context}\n\nQuestion: {query}\nAnswer:", max_length=700,
                             num_return_sequences=1)
        return response[0]['generated_text'].split("Answer:")[1].strip()
    elif rag_model == "Hugging Face (minilm-uncased)":
        nlp = hf_models["deepset/minilm-uncased-squad2"]
        response = nlp({'question': query, 'context': context})
        return response['answer']
