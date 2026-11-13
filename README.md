
# Jwalitha Profile AI Chatbot

An AI-powered personal profile chatbot built using
Retrieval-Augmented Generation (RAG).

## Technologies Used
- Python
- Streamlit
- Sentence Transformers
- FAISS
- Groq API
- GPT-OSS 120B
- PyPDF

## Architecture
PDF → Text Extraction → Chunking → Embeddings
→ FAISS Retrieval → Groq LLM → Answer

## Features
- Ask questions about professional experience
- Retrieve technical skills and expertise
- Explore workshops and training experience
- Context-based AI responses

## Run Locally
pip install -r requirements.txt
streamlit run app.py
