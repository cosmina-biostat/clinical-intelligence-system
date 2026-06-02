import streamlit as st
import tempfile, os
from src.rag.pdf_parser import extract_text_from_pdf, chunk_text
from src.rag.embedder import build_faiss_index
from src.rag.retriever import load_index, retrieve_context, ask_claude
from src.utils.config import MODELS_DIR

st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("Clinical Protocol Q&A (RAG + Claude)")

uploaded = st.file_uploader("Upload a clinical study protocol PDF", type="pdf")

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Parsing and indexing PDF..."):
        text   = extract_text_from_pdf(tmp_path)
        chunks = chunk_text(text)
        index, chunks = build_faiss_index(chunks, index_name="rag_upload")
    st.success(f"Indexed {len(chunks)} chunks from {uploaded.name}")
    os.unlink(tmp_path)

st.divider()
st.subheader("Ask a question about the loaded protocol")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

question = st.chat_input("e.g. What are the exclusion criteria?")

index_path = MODELS_DIR / "rag_upload.index"
if question:
    if not index_path.exists():
        st.error("No document indexed yet. Upload a PDF above first.")
    else:
        st.chat_message("user").write(question)
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Searching protocol and asking Claude..."):
            idx, chunks = load_index("rag_upload")
            context  = retrieve_context(question, idx, chunks)
            answer   = ask_claude(question, context)
        st.chat_message("assistant").write(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
