from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: Optional[str] = "You are a helpful AI assistant."
    messages: list[str]
    has_documents: bool = False

from ai_agent import get_response_from_ai_agent, initialize_rag

ALLOWED_MODELS = {
    "groq": ["llama-3.3-70b-versatile", "mixtral-8x7b-32768", "llama3-70b-8192"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

documents_uploaded = False

@app.get("/")
def read_root():
    return {"message": "LangGraph AI Agent with RAG", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "documents_uploaded": documents_uploaded}

@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    global documents_uploaded
    
    try:
        class FileWrapper:
            def __init__(self, file):
                self.name = file.filename
                self._content = None
                self._file = file
            
            def getvalue(self):
                if self._content is None:
                    import asyncio
                    self._content = asyncio.run(self._file.read())
                return self._content
        
        wrapped_files = [FileWrapper(f) for f in files]
        vector_store = initialize_rag(wrapped_files)
        
        if vector_store is None:
            return {"status": "error", "message": "No valid documents"}
        
        documents_uploaded = True
        return {
            "status": "success",
            "message": f"Processed {len(files)} document(s)",
            "files": [f.filename for f in files]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_endpoint(request: RequestState):
    global documents_uploaded
    
    try:
        if request.model_provider not in ALLOWED_MODELS:
            raise HTTPException(status_code=400, detail="Invalid provider")
        
        if request.model_name not in ALLOWED_MODELS[request.model_provider]:
            raise HTTPException(status_code=400, detail="Invalid model")
        
        if not request.messages or not request.messages[-1].strip():
            raise HTTPException(status_code=400, detail="Empty query")
        
        has_docs = documents_uploaded
        
        print(f"Documents uploaded: {documents_uploaded}")
        print(f"Has docs for agent: {has_docs}")
        
        response = get_response_from_ai_agent(
            request.model_name, 
            request.messages[-1], 
            request.system_prompt or "You are a helpful AI assistant.", 
            request.model_provider,
            has_documents=has_docs
        )
        
        return {
            "response": response,
            "model_used": request.model_name,
            "provider": request.model_provider,
            "rag_enabled": has_docs,
            "status": "success"
        }
    
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear-documents")
def clear_documents():
    global documents_uploaded
    from ai_agent import vector_store
    vector_store = None
    documents_uploaded = False
    return {"status": "success", "message": "Documents cleared"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting backend on http://127.0.0.1:8999")
    uvicorn.run(app, host="127.0.0.1", port=8999)