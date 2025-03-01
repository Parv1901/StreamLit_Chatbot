import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
import time
from mistralai import Mistral

# Set your API key here
API_KEY = st.secrets["MISTRAL_API_KEY"]

def fetch_text_from_urls(urls):
    all_chunks = []
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            tag = soup.find("div")
            if tag:
                text = tag.get_text(strip=True)
                chunk_size = 512
                chunks = [text[i: i + chunk_size] for i in range(0, len(text), chunk_size)]
                all_chunks.extend(chunks)
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to retrieve {url}. Error: {e}")
    return all_chunks

def get_text_embedding(list_txt_chunks):
    client = Mistral(api_key=API_KEY)
    max_tokens = 16000  # Adjusted for API limits
    embeddings = []
    for i in range(0, len(list_txt_chunks), max_tokens // 512):  # Process in batches
        batch = list_txt_chunks[i:i + (max_tokens // 512)]
        attempt = 0
        while attempt < 5:
            try:
                embeddings_batch_response = client.embeddings.create(model="mistral-embed", inputs=batch)
                embeddings.extend(embeddings_batch_response.data)
                break
            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff
                st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                attempt += 1
    return embeddings

def build_faiss_index(chunks):
    text_embeddings = get_text_embedding(chunks)
    embeddings = np.array([e.embedding for e in text_embeddings])
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, chunks

def retrieve_relevant_chunks(question, index, chunks, k=2):
    question_embedding = np.array([get_text_embedding([question])[0].embedding])
    D, I = index.search(question_embedding, k)
    return [chunks[i] for i in I[0]]

def generate_response(prompt):
    client = Mistral(api_key=API_KEY)
    messages = [{"role": "user", "content": prompt}]
    attempt = 0
    while attempt < 5:
        try:
            chat_response = client.chat.complete(model="mistral-large-latest", messages=messages)
            return chat_response.choices[0].message.content
        except Exception as e:
            wait_time = 2 ** attempt  # Exponential backoff
            st.warning(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            attempt += 1
    return "Failed to generate response after multiple attempts. Please try again later."

# Streamlit UI
st.title("UDST RAG Chatbot")

st.subheader("Available Policies:")
policies = [
    "Graduation Policy",
    "Graduate Admissions Policy",
    "Graduate Academic Standing Policy",
    "Graduate Academic Standing Procedure",
    "Graduate Final Grade Policy",
    "Graduate Final Grade Procedure",
    "Scholarship and Financial Assistance",
    "International Student Policy",
    "International Student Procedure",
    "Registration Policy"
]

for policy in policies:
    st.write(f"- {policy}")

urls = [
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-admissions-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-procedure",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-procedure",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/scholarship-and-financial-assistance",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-procedure",
    "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy"
]

with st.spinner("Fetching and processing policy documents..."):
    chunks = fetch_text_from_urls(urls)
    index, processed_chunks = build_faiss_index(chunks)

question = st.text_input("Ask a question about UDST policies:")
if question:
    with st.spinner("Retrieving relevant information..."):
        retrieved_chunks = retrieve_relevant_chunks(question, index, processed_chunks)
    
    prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunks}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {question}
    Answer:
    """
    
    with st.spinner("Generating response..."):
        response = generate_response(prompt)
    
    st.subheader("Response:")
    st.write(response)
