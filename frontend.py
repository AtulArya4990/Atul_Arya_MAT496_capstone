import streamlit as st
import requests

st.set_page_config(page_title="LangGraph AI Agent with RAG", layout="wide", page_icon="🤖")

st.title("🤖 LangGraph AI Agent with RAG")
st.write("Autonomous Intelligence + Document Understanding")

API_URL = "http://127.0.0.1:8999"

if 'documents_uploaded' not in st.session_state:
    st.session_state.documents_uploaded = False

with st.sidebar:
    st.header("Configuration")
    
    provider = st.radio("Provider:", ["groq", "openai"])
    
    if provider == "groq":
        models = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "llama3-70b-8192"]
    else:
        models = ["gpt-4o-mini", "gpt-4o"]
    
    selected_model = st.selectbox("Model:", models)
    
    system_prompt = st.text_area(
        "System Prompt:", 
        value="You are a helpful AI assistant.", 
        height=100
    )
    
    st.markdown("---")
    st.subheader("📚 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("📤 Upload", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    files = [("files", (file.name, file, file.type)) for file in uploaded_files]
                    response = requests.post(f"{API_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        st.success("✅ Documents uploaded!")
                        st.session_state.documents_uploaded = True
                    else:
                        st.error("❌ Upload failed")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    if st.session_state.documents_uploaded:
        st.success("✅ Documents loaded")
        if st.button("🗑️ Clear", use_container_width=True):
            requests.post(f"{API_URL}/clear-documents")
            st.session_state.documents_uploaded = False
            st.rerun()
    else:
        st.info("📄 No documents")

user_query = st.text_area(
    "Your Question:",
    height=200,
    placeholder="Ask anything..."
)

if st.button("Ask Agent", type="primary"):
    if user_query.strip():
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "model_name": selected_model,
                    "model_provider": provider,
                    "system_prompt": system_prompt,
                    "messages": [user_query],
                    "has_documents": st.session_state.documents_uploaded
                }
                
                response = requests.post(f"{API_URL}/chat", json=payload, timeout=90)
                
                if response.status_code == 200:
                    data = response.json()
                    st.subheader("Response:")
                    st.write(data.get("response"))
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model", data.get("model_used"))
                    with col2:
                        st.metric("Provider", data.get("provider"))
                    with col3:
                        rag = "✅" if data.get("rag_enabled") else "❌"
                        st.metric("RAG", rag)
                    
                    st.success("✅ Done")
                else:
                    st.error(f"Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Backend not running")
            except Exception as e:
                st.error(f"❌ {str(e)}")
    else:
        st.warning("Enter a question")