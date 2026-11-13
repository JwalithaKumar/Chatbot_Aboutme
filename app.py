
import os
import streamlit as st
import numpy as np
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# --------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Jwalitha's AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Jwalitha's AI Assistant")

st.write(
    "Hello! I am Jwalitha's AI Assistant. "
    "Ask me about her professional experience, "
    "technical skills, training programs and achievements."
)

st.divider()


# --------------------------------
# 2. GET GROQ API KEY
# --------------------------------

def get_api_key():

    key = os.getenv("GROQ_API_KEY")

    if not key:
        try:
            key = st.secrets["GROQ_API_KEY"]
        except Exception:
            key = None

    return key


# --------------------------------
# 3. TEXT CHUNKING
# --------------------------------

def create_chunks(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# --------------------------------
# 4. LOAD PDF AND BUILD FAISS INDEX
# --------------------------------

@st.cache_resource
def load_knowledge_base():

    # Read the fixed profile PDF
    reader = PdfReader("Jwalithapro.pdf")

    full_text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            full_text += page_text + "\n"

    if not full_text.strip():
        raise ValueError(
            "No readable text found in Jwalithapro.pdf"
        )

    # Split text into chunks
    chunks = create_chunks(full_text)

    if not chunks:
        raise ValueError("No text chunks were created.")

    # Load embedding model
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # Generate embeddings
    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    # Create FAISS index
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return embedding_model, chunks, index


# --------------------------------
# 5. LOAD KNOWLEDGE BASE
# --------------------------------

try:

    with st.spinner(
        "Loading Jwalitha's professional profile..."
    ):

        embedding_model, chunks, index = (
            load_knowledge_base()
        )

    st.success("Profile knowledge base loaded successfully!")

except Exception as e:

    st.error(f"Unable to load profile: {e}")

    st.stop()


# --------------------------------
# 6. CONNECT TO GROQ
# --------------------------------

api_key = get_api_key()

if not api_key:

    st.error(
        "Groq API key not found. "
        "Please configure GROQ_API_KEY in Streamlit Secrets."
    )

    st.stop()

client = Groq(api_key=api_key)


# --------------------------------
# 7. RAG FUNCTION
# --------------------------------

def ask_rag(question):

    # Convert user question into an embedding
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    # Normalize question embedding
    faiss.normalize_L2(query_embedding)

    # Search for the 3 most relevant chunks
    distances, indices = index.search(
        query_embedding,
        k=min(3, len(chunks))
    )

    # Collect retrieved context
    retrieved_context = ""

    for idx in indices[0]:

        if idx != -1:

            retrieved_context += (
                chunks[idx] + "\n\n"
            )

    # Prepare prompt
    prompt = f"""
You are Jwalitha's professional AI assistant.

Answer the user's question using ONLY the
information available in the profile context below.

Rules:
1. Do not invent facts or assume information.
2. Do not use outside knowledge to describe Jwalitha.
3. If the answer is not available in the profile,
   clearly say that the profile does not provide
   this information.
4. Give clear, professional and easy-to-understand answers.
5. Use bullet points when appropriate.

PROFILE CONTEXT:
{retrieved_context}

USER QUESTION:
{question}

ANSWER:
"""

    # Send prompt to Groq LLM
    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.2
    )

    return response.choices[0].message.content


# --------------------------------
# 8. CHAT HISTORY
# --------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------
# 9. USER INPUT AND CHAT RESPONSE
# --------------------------------

question = st.chat_input(
    "Ask something about Jwalitha..."
)

if question:

    # Display user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Finding the answer..."):

            try:

                answer = ask_rag(question)

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                st.error(
                    f"Something went wrong: {e}"
                )


# --------------------------------
# 10. SIDEBAR
# --------------------------------

with st.sidebar:

    st.header("About this Chatbot")

    st.write(
        "This AI assistant uses Jwalitha's "
        "professional profile as its knowledge source."
    )

    st.subheader("Technology Used")

    st.write("- Python")
    st.write("- Streamlit")
    st.write("- Sentence Transformers")
    st.write("- FAISS")
    st.write("- Groq API")
    st.write("- GPT-OSS 120B")
    st.write("- Retrieval-Augmented Generation (RAG)")

    st.divider()

    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()
