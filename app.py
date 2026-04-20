import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 PDF Chatbot ")
st.subheader("Github->ghost-bw")
st.write("MODEL:", "llama-3.1-8b-instant")

# =========================
# API KEY
# =========================
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("GROQ API key not found in secrets.toml")
    st.stop()

if not groq_api_key:
    st.error("❌ GROQ API key not found.")
    st.stop()

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# =========================
# Upload PDFs
# =========================
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files and st.session_state.vectorstore is None:

    documents = []

    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())

        loader = PyPDFLoader(file.name)
        loaded_docs = loader.load()

        if loaded_docs:
            documents.extend(loaded_docs)

    # ❌ No text extracted
    if not documents:
        st.error("❌ Could not extract text from PDFs")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # ❌ No chunks created
    if not docs:
        st.error("❌ No chunks created from documents")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ❌ Prevent FAISS crash
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"❌ Vector DB creation failed: {e}")
        st.stop()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    st.session_state.vectorstore = retriever
    st.success("✅ PDFs processed. You can now chat!")

# =========================
# LLM
# =========================
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant"
)

# =========================
# Chat UI
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask something from your PDFs...")

if query and st.session_state.vectorstore:

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    # retrieve context safely
    try:
        docs = st.session_state.vectorstore.invoke(query)
    except Exception as e:
        st.error(f"❌ Retrieval error: {e}")
        st.stop()

    if not docs:
        st.warning("⚠️ No relevant context found.")
        st.stop()

    context = "\n\n".join([doc.page_content for doc in docs])

    history = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-5:]]
    )

    prompt = f"""
You are a helpful AI assistant.

Use ONLY the context below to answer.
If answer is not in context, say "I don't know".

Chat History:
{history}

Context:
{context}

Question:
{query}
"""

    try:
        response = llm.invoke(prompt)
        answer = response.content
    except Exception as e:
        st.error(f"❌ LLM error: {e}")
        st.stop()

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.write(answer)

elif query:
    st.warning("⚠️ Please upload PDFs first.")
