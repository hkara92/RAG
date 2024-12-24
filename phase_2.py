# Streamlit chat app with Groq integration
import warnings
import logging

import streamlit as st

# Import Groq chat model and prompt tools
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Turn off warnings and extra logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.title('Ask Chatbot!')
# Initialize chat history storage
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Show previous chat messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Get user input
prompt = st.chat_input('Pass your prompt here')

if prompt:
    # Display user's message
    st.chat_message('user').markdown(prompt)
    # Save user's message
    st.session_state.messages.append({'role':'user', 'content': prompt})
    
    # Prepare system prompt for Groq
    groq_sys_prompt = ChatPromptTemplate.from_template("""You are very smart at everything, you always give the best, 
                                            the most accurate and most precise answers. Answer the following Question: {user_prompt}.
                                            Start the answer directly. No small talk please""")

    # Choose a model for Groq
    # model = "mixtral-8x7b-32768"
    model="llama3-8b-8192"

    groq_chat = ChatGroq(
            groq_api_key=os.environ.get("GROQ_API_KEY"), 
            model_name=model
    )

    # Create a chain: system prompt -> Groq model -> string parser
    chain = groq_sys_prompt | groq_chat | StrOutputParser()
    # Run the chain with user's prompt
    response = chain.invoke({"user_prompt": prompt})

    # Display assistant's response
    st.chat_message('assistant').markdown(response)
    # Save assistant's response
    st.session_state.messages.append(
            {'role':'assistant', 'content':response})
    



