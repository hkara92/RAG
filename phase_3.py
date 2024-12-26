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

