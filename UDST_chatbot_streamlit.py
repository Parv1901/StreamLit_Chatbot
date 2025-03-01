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

# Custom CSS for a more beautiful design
# Custom CSS for a more beautiful design
st.markdown(
    """
    <style>
    /* Set all text to black */
    body, h1, h2, h3, h4, h5, h6, p, a, .stMarkdown, .stTextInput, .stButton, .stSpinner {
        color: black !important;
    }

    /* Ensure the background has a beautiful blue gradient */
    .stApp {
        background: linear-gradient(135deg, #a1c4fd, #c2e9fb, #e0c3fc) !important;
        background-size: 200% 200%;
        animation: gradientAnimation 10s ease infinite;
    }

    /* Gradient animation */
    @keyframes gradientAnimation {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Header styling */
    .stMarkdown h1 {
        color: #2e86c1 !important;
        font-family: 'Arial', sans-serif;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 20px;
        animation: fadeIn 2s ease-in-out;
    }

    /* Subheader styling */
    .stMarkdown h2 {
        color: #1a5276 !important;
        font-family: 'Arial', sans-serif;
        font-size: 1.8rem;
        margin-top: 20px;
        animation: slideIn 1.5s ease-in-out;
    }

    /* Fade-in animation */
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }

    /* Slide-in animation */
    @keyframes slideIn {
        0% { transform: translateX(-100%); opacity: 0; }
        100% { transform: translateX(0); opacity: 1; }
    }

    /* Hyperlink styling */
    a {
        color: #2e86c1 !important;
        text-decoration: none;
        font-weight: bold;
        transition: color 0.3s ease;
    }
    a:hover {
        color: #1a5276 !important;
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
        transition: background-color 0.3s ease, transform 0.3s ease;
    }
    .stButton button:hover {
        background-color: #1a5276 !important;
        transform: scale(1.05);
    }

    /* Input field styling */
    .stTextInput input {
        border-radius: 25px;
        padding: 10px;
        border: 1px solid #2e86c1;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: box-shadow 0.3s ease;
    }
    .stTextInput input:focus {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    /* Response box styling */
    .stTextArea textarea {
        border-radius: 15px;
        padding: 15px;
        border: 1px solid #2e86c1;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: box-shadow 0.3s ease;
    }
    .stTextArea textarea:focus {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    /* Add some padding and spacing */
    .stMarkdown, .stTextInput, .stButton, .stSpinner, .stTextArea {
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Title and description
st.title("RAG Chatbot for UDST Policies!")
st.markdown("Welcome to the UDST Policy RAG Chatbot! Ask any questions about UDST policies here!.")

# List of policies with hyperlinks
st.subheader("Available Policies / The 10 policies the chatbot will answer questions about:")
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
question = st.text_input("Ask a Question:")

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
