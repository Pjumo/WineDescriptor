import os
import logging

import streamlit as st
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain import hub
from dotenv import load_dotenv
from langchain_google_vertexai import GemmaChatLocalHF
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


@st.cache_data
def parse_data():
    loader = CSVLoader(file_path='./wine-raitngs.csv', encoding='utf-8')
    return loader.load()


@st.cache_resource
def get_retriever():
    try:
        st.write("Embedding !!!!!!!!!!!!!!")
        model_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embedding = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        st.write("CSV Loading !!!!!!!!!!!!!!\n")
        data = parse_data()
        st.write("Pinecone Vector Store !!!!!!!!!!!!!!\n")
        database = PineconeVectorStore.from_documents(data, embedding, index_name='wine-index')
        retriever = database.as_retriever()
        return retriever
    except Exception as e:
        logging.error(f"Error creating retriever: {e}")
        raise


@st.cache_resource
def get_llm():
    st.write("Model Downloading !!!!!!!!!!!!!!\n")
    return GemmaChatLocalHF(model_name="google/gemma-2b", hf_access_token=os.getenv("HF_ACCESS_TOKEN"))


@st.cache_resource
def get_rag_chain():
    prompt = hub.pull("rlm/rag-prompt")
    llm = get_llm()
    retriever = get_retriever()
    st.write("QA Chain !!!!!!!!!!!!!!\n")
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    st.write("History Chain !!!!!!!!!!!!!!\n")
    conversational_chain = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    ).pick('answer')

    return conversational_chain


@st.cache_data
def get_ai_message(user_message):
    load_dotenv(verbose=True)
    try:
        rag_chain = get_rag_chain()
        st.write("Streaming !!!!!!!!!!!!!!\n")
        ai_message = rag_chain.stream(
            {
                "input": user_message
            },
            config={
                "configurable": {"session_id": "abc123"}
            })
        return ai_message
    except Exception as e:
        logging.error(f"Error creating AI message: {e}")
        return "An error occurred while processing your request."
