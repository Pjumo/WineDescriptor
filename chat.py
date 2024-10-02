import streamlit as st
from llm import get_ai_message

st.set_page_config(page_title="Wine Descriptor", page_icon="🍷")
st.title("🍷 Wine Descriptor")
st.caption("Answering your questions about wine")

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
        ai_message = get_ai_message(user_question)
        with st.chat_message('ai'):
            st.write_stream(ai_message)
        st.session_state.message_list.append({'role': 'ai', 'content': ai_message})
