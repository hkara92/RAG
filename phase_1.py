# Part 1: Initialize Streamlit user interface for chat application
import warnings
import logging

import streamlit as st

# Suppress warnings and transformer info logs for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Set the title of the app
st.title('Ask Chatbot!')

# Initialize session state to store chat history if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Render all previous messages from session state
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Prompt user for input
prompt = st.chat_input('Pass your prompt here')

if prompt:
    # Display user's message in the chat window
    st.chat_message('user').markdown(prompt)
    # Append user prompt to the session history
    st.session_state.messages.append({'role':'user', 'content': prompt})
    # Generate a placeholder assistant response
    response = "I am your assistant"
    # Display assistant's response in the chat window
    st.chat_message('assistant').markdown(response)
    # Append assistant response to the session history
    st.session_state.messages.append(
            {'role':'assistant', 'content':response})


