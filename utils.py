import json
import os
import zipfile

import torch
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import cohere
from transformers import pipeline,AutoModelForCausalLM, AutoTokenizer

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