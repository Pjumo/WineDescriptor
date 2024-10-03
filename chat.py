import streamlit as st
from llm import get_ai_response
from dotenv import load_dotenv
import os

st.set_page_config(page_title="Wine Descriptor", page_icon="🍷")
st.title("🍷 Wine Descriptor")
st.caption("Answering your questions about wine")

load_dotenv(verbose=True)
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])


if user_question := st.chat_input(placeholder="Ask me about wine!"):
    with st.chat_message('user'):
        st.write(user_question)
    st.session_state.message_list.append({'role': 'user', 'content': user_question})

    with st.spinner("Generating answer..."):
        ai_response = get_ai_response(user_question)
        with st.chat_message('ai'):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({'role': 'ai', 'content': ai_message})
