UDST RAG Chatbot
This Streamlit application is a Retrieval-Augmented Generation (RAG) chatbot built for answering questions based on official UDST (University of Doha for Science and Technology) policy documents. It fetches data from UDST's policy webpages, indexes it using FAISS, and uses the Mistral API to generate high-quality answers.

Features:
Fetches text from multiple UDST policy URLs
Chunks and embeds content using Mistral’s embedding model
Builds a FAISS index for semantic search
Retrieves the most relevant chunks to the user's query
Uses Mistral's large language model to answer based on retrieved context
Stylish and responsive UI using custom CSS in Streamlit

Technologies Used:
- Streamlit for the user interface
- BeautifulSoup for scraping webpage content
- FAISS for fast similarity search
- Mistral AI for embeddings and chat completions
- Python, NumPy, Requests, and Time libraries

