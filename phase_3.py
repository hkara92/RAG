# Import basic libraries
import os
import warnings
import logging

import streamlit as st

# Import libraries for chatting model
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Import libraries for document processing and search
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# Turn off warning and info messages
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title('Ask Chatbot!')

# Remember past chat messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Show previous messages in chat
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Prepare document index (only once)
@st.cache_resource
def get_vectorstore():
    pdf_name = "./reflexion.pdf"
    loaders = [PyPDFLoader(pdf_name)]
    # Split PDF into chunks and build a searchable index
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(loaders)
    return index.vectorstore

prompt = st.chat_input('Pass your prompt here')

if prompt:
    # Show user's message
    st.chat_message('user').markdown(prompt)
    # Save user's message
    st.session_state.messages.append({'role':'user', 'content': prompt})
    
    # Prepare system prompt for the model
    groq_sys_prompt = ChatPromptTemplate.from_template(
        "You are very smart at everything, you always give the best, "
        "the most accurate and most precise answers. Answer the following Question: {user_prompt}. "
        "Start the answer directly. No small talk please"
    )

    model="llama3-8b-8192"
    # Connect to the GROQ API
    groq_chat = ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"), 
        model_name=model
    )

    # Use the index to answer the question
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("Failed to load document")
      
        chain = RetrievalQA.from_chain_type(
            llm=groq_chat,
            chain_type='stuff',
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True
        )
        result = chain({"query": prompt})
        response = result["result"]  # Only keep the answer
        # Old method: get_response_from_groq(prompt)
        st.chat_message('assistant').markdown(response)
        # Save assistant's reply
        st.session_state.messages.append({'role':'assistant', 'content':response})
    except Exception as e:
        # Show error if something goes wrong
        st.error(f"Error: {str(e)}")
