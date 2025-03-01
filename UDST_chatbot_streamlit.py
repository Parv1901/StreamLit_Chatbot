import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
import time
from mistralai import Mistral

# Set your API key here
API_KEY = st.secrets["MISTRAL_API_KEY"]

# --- Functions ---
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
            time.sleep(wait_time)
            attempt += 1
    return "Failed to generate response after multiple attempts. Please try again later."

# --- Streamlit UI ---
st.set_page_config(page_title="UDST RAG Chatbot", page_icon="📚", layout="centered")

# Custom CSS for creative design
st.markdown(
    """
    <style>
    /* Set all text to black */
    body, h1, h2, h3, h4, h5, h6, p, a, .stMarkdown, .stTextInput, .stButton, .stSpinner {
        color: black !important;
    }

    /* Ensure the background is light with a gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2) !important;
    }

    /* Header styling */
    .stMarkdown h1 {
        color: #2e86c1 !important;
        font-family: 'Arial', sans-serif;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 20px;
    }

    /* Subheader styling */
    .stMarkdown h2 {
        color: #1a5276 !important;
        font-family: 'Arial', sans-serif;
        font-size: 1.8rem;
        margin-top: 20px;
    }

    /* Hyperlink styling */
    a {
        color: #2e86c1 !important;
        text-decoration: none;
        font-weight: bold;
    }
    a:hover {
        text-decoration: underline;
    }

    /* Button styling */
    .stButton button {
        background-color: #2e86c1 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 25px;
        padding: 10px 20px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #1a5276 !important;
    }

    /* Input field styling */
    .stTextInput input {
        border-radius: 25px;
        padding: 10px;
        border: 1px solid #2e86c1;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Spinner styling */
    .stSpinner {
        color: #2e86c1 !important;
    }

    /* Response box styling */
    .stTextArea textarea {
        border-radius: 15px;
        padding: 15px;
        border: 1px solid #2e86c1;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.title("UDST RAG Chatbot")
st.markdown("Welcome to the UDST Policy Chatbot! Ask questions about UDST policies, and get instant answers.")

# List of policies with hyperlinks
st.subheader("Available Policies:")
policies = {
    "Graduation Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
    "Graduate Admissions Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-admissions-policy",
    "Graduate Academic Standing Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-policy",
    "Graduate Academic Standing Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-procedure",
    "Graduate Final Grade Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-policy",
    "Graduate Final Grade Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-procedure",
    "Scholarship and Financial Assistance": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/scholarship-and-financial-assistance",
    "International Student Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy",
    "International Student Procedure": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-procedure",
    "Registration Policy": "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy"
}

# Display policies as clickable hyperlinks
for policy, url in policies.items():
    st.markdown(f"- [{policy}]({url})", unsafe_allow_html=True)

# Text input for user query
question = st.text_input("Ask a question about UDST policies:")

# Button to submit query
if st.button("Submit"):
    if question:
        with st.spinner("Fetching and processing policy documents..."):
            chunks = fetch_text_from_urls(list(policies.values()))
            index, processed_chunks = build_faiss_index(chunks)
        
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
        st.text_area("", value=response, height=200)  # Response box with custom styling
    else:
        st.warning("Please enter a question.")
