import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import faiss
from mistralai import Mistral, UserMessage
import time

# --- Functions ---

# Function to scrape policy text
def scrape_policy_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        html_doc = response.text
        soup = BeautifulSoup(html_doc, "html.parser")
        tag = soup.find("div")
        if tag:
            return tag.get_text(strip=True)
        else:
            return None
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve {url}. Error: {e}")
        return None

# Function to generate embeddings with rate limiting
def get_text_embedding(list_txt_chunks, api_key, batch_size=50, delay=1):
    client = Mistral(api_key=api_key)
    embeddings = []
    
    # Process chunks in smaller batches
    for i in range(0, len(list_txt_chunks), batch_size):
        batch = list_txt_chunks[i:i + batch_size]
        
        # Make API request
        try:
            embeddings_batch_response = client.embeddings.create(
                model="mistral-embed",
                inputs=batch
            )
            embeddings.extend(embeddings_batch_response.data)
        except Exception as e:
            print(f"Error processing batch: {e}")
            continue
        
        # Introduce a delay to avoid hitting rate limits
        time.sleep(delay)
    
    return embeddings

# Function to query the RAG model
def query_rag_model(question, chunks, embeddings, api_key):
    question_embeddings = np.array([get_text_embedding([question], api_key)[0].embedding])
    D, I = index.search(question_embeddings, k=2)
    retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
    prompt = f"""
    Context information is below.
    ---------------------
    {retrieved_chunk}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {question}
    Answer:
    """
    return mistral(prompt, api_key)

# Function to call Mistral API
def mistral(user_message, api_key, model="mistral-large-latest"):
    client = Mistral(api_key=api_key)
    messages = [UserMessage(content=user_message)]
    chat_response = client.chat.complete(
        model=model,
        messages=messages,
    )
    return chat_response.choices[0].message.content

# --- Streamlit App Code ---

# --- Streamlit App Code ---

# Title of the app
st.title("UDST Policy Chatbot")

# List of policies and their corresponding URLs
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

# Display the list of policies as hyperlinks
st.write("List of Policies:")
for policy, url in policies.items():
    st.markdown(f"- [{policy}]({url})", unsafe_allow_html=True)

# Text input for user query
user_query = st.text_input("Enter your query:")

# Button to submit query
if st.button("Submit"):
    if user_query:
        # Scrape and process policy texts
        policy_texts = []
        for url in policies.values():
            text = scrape_policy_text(url)
            if text:
                policy_texts.append(text)

        # Split text into chunks
        chunk_size = 512
        chunks = [text[i:i + chunk_size] for text in policy_texts for i in range(0, len(text), chunk_size)]

        # Generate embeddings with rate limiting
        api_key = "i7oz9ULpeKEN5gafBm4WjCnB3HlZvRKA"  # Replace with your actual API key
        text_embeddings = get_text_embedding(chunks, api_key, batch_size=50, delay=1)  # Adjust delay as needed

        # Create FAISS index
        d = len(text_embeddings[0].embedding)
        index = faiss.IndexFlatL2(d)
        embeddings = np.array([text_embeddings[i].embedding for i in range(len(text_embeddings))])
        index.add(embeddings)

        # Call your RAG model function here
        answer = query_rag_model(user_query, chunks, embeddings, api_key)
        st.text_area("Answer:", value=answer, height=200)
    else:
        st.warning("Please enter a query.")




# WORKING

# import streamlit as st
# import requests
# from bs4 import BeautifulSoup
# import numpy as np
# import faiss
# from mistralai import Mistral, UserMessage
# import time

# # --- Functions ---

# # Function to scrape policy text
# def scrape_policy_text(url):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         html_doc = response.text
#         soup = BeautifulSoup(html_doc, "html.parser")
#         tag = soup.find("div")
#         if tag:
#             return tag.get_text(strip=True)
#         else:
#             return None
#     except requests.exceptions.RequestException as e:
#         print(f"Failed to retrieve {url}. Error: {e}")
#         return None

# # Function to generate embeddings with rate limiting
# def get_text_embedding(list_txt_chunks, api_key, batch_size=50, delay=1):
#     client = Mistral(api_key=api_key)
#     embeddings = []
    
#     # Process chunks in smaller batches
#     for i in range(0, len(list_txt_chunks), batch_size):
#         batch = list_txt_chunks[i:i + batch_size]
        
#         # Make API request
#         try:
#             embeddings_batch_response = client.embeddings.create(
#                 model="mistral-embed",
#                 inputs=batch
#             )
#             embeddings.extend(embeddings_batch_response.data)
#         except Exception as e:
#             print(f"Error processing batch: {e}")
#             continue
        
#         # Introduce a delay to avoid hitting rate limits
#         time.sleep(delay)
    
#     return embeddings

# # Function to query the RAG model
# def query_rag_model(question, chunks, embeddings, api_key):
#     question_embeddings = np.array([get_text_embedding([question], api_key)[0].embedding])
#     D, I = index.search(question_embeddings, k=2)
#     retrieved_chunk = [chunks[i] for i in I.tolist()[0]]
#     prompt = f"""
#     Context information is below.
#     ---------------------
#     {retrieved_chunk}
#     ---------------------
#     Given the context information and not prior knowledge, answer the query.
#     Query: {question}
#     Answer:
#     """
#     return mistral(prompt, api_key)

# # Function to call Mistral API
# def mistral(user_message, api_key, model="mistral-large-latest"):
#     client = Mistral(api_key=api_key)
#     messages = [UserMessage(content=user_message)]
#     chat_response = client.chat.complete(
#         model=model,
#         messages=messages,
#     )
#     return chat_response.choices[0].message.content

# # --- Streamlit App Code ---

# # Title of the app
# st.title("UDST Policy Chatbot")

# # List of policies
# policies = [
#     "Graduation Policy",
#     "Graduate Admissions Policy",
#     "Graduate Academic Standing Policy",
#     "Graduate Academic Standing Procedure",
#     "Graduate Final Grade Policy",
#     "Graduate Final Grade Procedure",
#     "Scholarship and Financial Assistance",
#     "International Student Policy",
#     "International Student Procedure",
#     "Registration Policy"
# ]

# # Dropdown to select policy
# selected_policy = st.selectbox("Select a Policy", policies)

# # Text input for user query
# user_query = st.text_input("Enter your query:")

# # Button to submit query
# if st.button("Submit"):
#     if user_query:
#         # List of policy URLs
#         urls = [
#              "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduation-policy",
#              "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-admissions-policy",
#              "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-policy",
#              "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/graduate-academic-standing-procedure",
#              "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-policy",
#              "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/graduate-final-grade-procedure",
#              "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/scholarship-and-financial-assistance",
#              "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-policy",
#              "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/udst-policies-and-procedures/international-student-procedure",
#              "https://www.udst.edu.qa/about-udst/institutional-excellence-ie/policies-and-procedures/registration-policy"
#          ]

#         # Scrape and process policy texts
#         policy_texts = []
#         for url in urls:
#             text = scrape_policy_text(url)
#             if text:
#                 policy_texts.append(text)

#         # Split text into chunks
#         chunk_size = 512
#         chunks = [text[i:i + chunk_size] for text in policy_texts for i in range(0, len(text), chunk_size)]

#         # Generate embeddings with rate limiting
#         api_key = "i7oz9ULpeKEN5gafBm4WjCnB3HlZvRKA"  # Replace with your actual API key
#         text_embeddings = get_text_embedding(chunks, api_key, batch_size=50, delay=1)  # Adjust delay as needed

#         # Create FAISS index
#         d = len(text_embeddings[0].embedding)
#         index = faiss.IndexFlatL2(d)
#         embeddings = np.array([text_embeddings[i].embedding for i in range(len(text_embeddings))])
#         index.add(embeddings)

#         # Call your RAG model function here
#         answer = query_rag_model(user_query, chunks, embeddings, api_key)
#         st.text_area("Answer:", value=answer, height=200)
#     else:
#         st.warning("Please enter a query.")


