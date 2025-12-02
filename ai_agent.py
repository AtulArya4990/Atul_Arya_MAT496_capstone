import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import tempfile

vector_store = None

def initialize_rag(uploaded_files):
    global vector_store
    
    if not uploaded_files:
        return None
    
    documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            if uploaded_file.name.endswith('.pdf'):
                loader = PyPDFLoader(tmp_path)
            elif uploaded_file.name.endswith('.txt'):
                loader = TextLoader(tmp_path)
            else:
                continue
            
            docs = loader.load()
            documents.extend(docs)
        finally:
            os.unlink(tmp_path)
    
    if not documents:
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(splits, embeddings)
    
    return vector_store