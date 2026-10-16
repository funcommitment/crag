import streamlit as st
import tempfile
from loaders import load_pdf
from embeddings import hybrid_search
from main import run_crag_agent
import os
# UI Configuration
st.set_page_config(page_title="CRAG Document Explorer", page_icon="🔍", layout="centered")
st.title("🔍 CRAG Document Verification Agent")

# ==========================================
# 1. SESSION STATE & DATABASE INITIALIZATION
# ==========================================
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Document Setup")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file:
        # Check if it's a NEW file so users can swap PDFs seamlessly
        if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
            
            with st.status("Building Hybrid Search Database...", expanded=True) as status:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                status.write("Extracting and chunking text...")
                splitted_chunks = load_pdf(tmp_path)
                
                # CRITICAL: Delete the temp file to prevent server disk crashes
                os.remove(tmp_path) 
                
                status.write("Building BM25 and ChromaDB indexes...")
                bm25_index, texts, metadatas, ids, collection, embedding_model = hybrid_search(splitted_chunks)
                
                st.session_state.bm25_index = bm25_index
                st.session_state.texts = texts
                st.session_state.metadatas = metadatas
                st.session_state.ids = ids
                st.session_state.collection = collection
                st.session_state.embedding_model = embedding_model
                st.session_state.db_ready = True
                
                # Track the current file name so we know if they upload a different one
                st.session_state.current_file = uploaded_file.name 
                
                status.update(label="Database Ready!", state="complete", expanded=False)

# ==========================================
# 2. CHAT INTERFACE
# ==========================================
# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
if user_query := st.chat_input("Ask a question..."):
    if "db_ready" not in st.session_state:
        st.error("Please upload a PDF in the sidebar first.")
        st.stop()

    # Display user query
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Agent Execution
    with st.chat_message("assistant"):
        # The st.status container is passed directly into your backend function
        with st.status("Agent analyzing text...", expanded=True) as status:
            result = run_crag_agent(
                user_query=user_query,
                bm25_index=st.session_state.bm25_index,
                texts=st.session_state.texts,
                metadatas=st.session_state.metadatas,
                ids=st.session_state.ids,
                collection=st.session_state.collection,
                embedding_model=st.session_state.embedding_model,
                max_retries=3,
                ui=status  # Injecting the Streamlit UI container here
            )
        
        # Display Final Output
        if result:
            final_answer, sources = result
            st.markdown(final_answer)
            
            with st.expander("View Source Documents"):
                for idx, source in enumerate(sources):
                    st.info(f"**Chunk {idx+1}:**\n\n{source}")
                    
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
        else:
            fallback = "I have searched the index 3 times, but this specific provision does not appear to be explicitly mentioned. I cannot provide an answer without hallucinating."
            st.warning(fallback)
            st.session_state.messages.append({"role": "assistant", "content": fallback})