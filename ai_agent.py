import os
from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, TypedDict, Sequence
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
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

@tool
def search_documents(query: str) -> str:
    """Search uploaded documents. ALWAYS use this tool when user asks about uploaded files, documents, or their content."""
    global vector_store
    
    if vector_store is None:
        return "No documents uploaded."
    
    try:
        docs = vector_store.similarity_search(query, k=3)
        
        if not docs:
            return "No relevant information found in documents."
        
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(f"Excerpt {i}:\n{doc.page_content}\n")
        
        return "\n".join(results)
    except Exception as e:
        return f"Error: {str(e)}"

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def call_model(state: AgentState, llm, tools):
    messages = state["messages"]
    
    if tools:
        llm_with_tools = llm.bind_tools(tools)
        response = llm_with_tools.invoke(messages)
    else:
        response = llm.invoke(messages)
    
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"

def create_langgraph_agent(llm, system_prompt, has_documents=False):
    tools = [TavilySearchResults(max_results=3)]
    
    if has_documents:
        tools.append(search_documents)
    
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", lambda state: call_model(state, llm, tools))
    tool_node = ToolNode(tools)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    workflow.add_edge("tools", "agent")
    app = workflow.compile()
    
    return app

def get_response_from_ai_agent(llm_id, query, system_prompt, provider, has_documents=False):
    print(f"Agent called with has_documents={has_documents}")
    
    if provider == "groq":
        llm = ChatGroq(model=llm_id, temperature=0)
    elif provider == "openai":
        llm = ChatOpenAI(model=llm_id, temperature=0)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    agent = create_langgraph_agent(llm, system_prompt, has_documents)
    
    tool_instructions = """
You have access to a web search tool for current information."""
    
    if has_documents:
        tool_instructions += """
IMPORTANT: You also have a 'search_documents' tool to search uploaded documents.
When the user asks about uploaded files or documents, you MUST use the search_documents tool first."""
    
    enhanced_system_prompt = f"""{system_prompt}

{tool_instructions}

Choose the right tool based on the query."""
    
    messages = [
        SystemMessage(content=enhanced_system_prompt),
        HumanMessage(content=query)
    ]
    
    initial_state = {"messages": messages}
    result = agent.invoke(initial_state)
    final_messages = result.get("messages", [])
    
    for message in reversed(final_messages):
        if isinstance(message, AIMessage):
            return message.content
    
    return "No response generated"