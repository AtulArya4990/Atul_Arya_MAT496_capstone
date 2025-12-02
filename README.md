Template for creating and submitting MAT496 capstone project.

# Overview of MAT496

In this course, we have primarily learned Langgraph. This is helpful tool to build apps which can process unstructured `text`, find information we are looking for, and present the format we choose. Some specific topics we have covered are:

- Prompting
- Structured Output 
- Semantic Search
- Retreaval Augmented Generation (RAG)
- Tool calling LLMs & MCP
- Langgraph: State, Nodes, Graph

We also learned that Langsmith is a nice tool for debugging Langgraph codes.

------

# Capstone Project objective

The first purpose of the capstone project is to give a chance to revise all the major above listed topics. The second purpose of the capstone is to show your creativity. Think about all the problems which you can not have solved earlier, but are not possible to solve with the concepts learned in this course. For example, We can use LLM to analyse all kinds of news: sports news, financial news, political news. Another example, we can use LLMs to build a legal assistant. Pretty much anything which requires lots of reading, can be outsourced to LLMs. Let your imagination run free.


-------------------------

# Project report Template

## Title: Intelligent Multi-Tool AI Agent with Autonomous Decision Making

## Overview

This project implements an intelligent AI agent that autonomously decides which tools to use based on user queries. The agent integrates three powerful capabilities:

 1 Web Search - For current events, real-time data, and recent information
 2 RAG (Retrieval Augmented Generation) - For searching through uploaded documents (PDFs and text files)
 3 Direct Knowledge - For general questions that don't require external tools

The system uses LangGraph to create a stateful workflow where the LLM intelligently routes queries to appropriate tools without manual intervention. Users can interact through a Streamlit frontend that connects to a FastAPI backend, which orchestrates the LangGraph agent.

## Reason for picking up this project

This project aligns perfectly with the MAT496 course content by incorporating all major topics:

 ->Prompting: System prompts guide the agent's behavior and tool selection strategy
 ->Structured Output: The agent processes and formats responses from multiple sources
 ->Semantic Search: FAISS vector store enables similarity-based document retrieval
 ->RAG (Retrieval Augmented Generation): Documents are chunked, embedded, and retrieved based on query relevance
 ->Tool Calling LLMs: The agent autonomously decides when to call web search or document search tools
 ->LangGraph: Complete implementation with custom state management, nodes, conditional edges, and workflow compilation

The creativity lies in building an autonomous decision-making system where the LLM acts as an intelligent router, eliminating the need for manual tool selection. This mimics real-world AI assistants that seamlessly switch between different information sources.

## Video Summary Link: 

Make a short -  3-5 min video of yourself, put it on youtube/googledrive, and put its link in your README.md.

- you can use this free tool for recording https://screenrec.com/
- Video format should be like this:
- your face should be visible
- State the overall job of your agent: what inputs it takes, and what output it gives.
- Very quickly, explain how your agent acts on the input and spits out the output. 
- show an example run of the agent in the video


## Plan

I plan to execute these steps to complete my project.

[DONE] Step 1: Implement RAG Pipeline Foundation (ai_agent.py) - Set up FAISS vector store with HuggingFace embeddings, implement document loading for PDFs and text files, create text chunking with RecursiveCharacterTextSplitter, and build the initialize_rag function to process uploaded documents
[DONE] Step 2: Create Custom Tools for Agent (ai_agent.py) - Define the search_documents tool using @tool decorator for RAG functionality, integrate TavilySearchResults for web search capability, write clear tool descriptions to guide LLM decision-making, and implement error handling for both tools
[TODO] Step 3: Build LangGraph Workflow Structure (ai_agent.py) - Define AgentState with annotated message sequences, create call_model node function for LLM invocation with tool binding, implement should_continue conditional function for routing logic, and set up the StateGraph with proper entry points and edges
[TODO] Step 4: Implement Autonomous Agent Logic (ai_agent.py) - Create the create_langgraph_agent function to compile the workflow, write enhanced system prompts for intelligent tool selection, implement get_response_from_ai_agent as the main entry point, and add logic for the LLM to autonomously choose between web search, RAG, or direct answers
[TODO] Step 5: Develop FastAPI Backend (backend.py) - Create RESTful API endpoints for chat, document upload, and health checks, implement CORS middleware for frontend communication, handle document processing and vector store initialization, and manage global state for document availability tracking
[TODO] Step 6: Create Streamlit Frontend (frontend.py) - Build an intuitive user interface with model selection and configuration options, implement document upload functionality with drag-and-drop support, create real-time chat interface with response display, and add visual indicators for RAG status and tool usage

## Conclusion:

I had planned to achieve {this this}. I think I have/have-not achieved the conclusion satisfactorily. The reason for your satisfaction/unsatisfaction.

----------

# Added instructions:

- This is a `solo assignment`. Each of you will work alone. You are free to talk, discuss with chatgpt, but you are responsible for what you submit. Some students may be called for viva. You should be able to each and every line of work submitted by you.

- `commit` History maintenance.
  - Fork this repository and build on top of that.
  - For every step in your plan, there has to be a commit.
  - Change [TODO] to [DONE] in the plan, before you commit after that step. 
  - The commit history should show decent amount of work spread into minimum two dates. 
  - **All the commits done in one day will be rejected**. Even if you are capable of doing the whole thing in one day, refine it in two days.  
 
 - Deadline: Dec 2nd, Tuesday 11:59 pm


# Grading: total 25 marks

- Coverage of most of topics in this class: 20
- Creativity: 5
  