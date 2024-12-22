# Chatbot-with-RAG

A Streamlit-based chatbot application demonstrating progressive development through three phases, culminating in a Retrieval-Augmented Generation (RAG) system that leverages custom PDF data sources.

## Features

- **Phase 1: Basic UI**  
  - Sets up a minimal Streamlit chat interface  
  - Echoes a static response for user inputs

- **Phase 2: LLM Integration**  
  - Integrates with Groq’s ChatGroq API via LangChain  
  - Utilizes a system prompt template for concise, precise answers  
  - Parses and displays dynamic responses from the LLM

- **Phase 3: Retrieval-Augmented Generation**  
  - Loads and splits a PDF document (`reflexion.pdf`) into chunks  
  - Creates a vectorstore (using HuggingFace embeddings and Chromadb)  
  - Implements a `RetrievalQA` chain to answer queries with source retrieval  
  - Caches the vectorstore for performance with Streamlit’s `@st.cache_resource`

## Prerequisites

- **Python**: 3.8 or higher  
- **Streamlit**: for the web UI  
- **Pipenv** (recommended) or **pip**  
- **GROQ_API_KEY**: an environment variable pointing to your Groq API key  
- **PDF file**: Place `reflexion.pdf` in the project root (Phase 3)

