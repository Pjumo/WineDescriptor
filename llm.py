import os
import logging

from langchain_community.llms import HuggingFaceTextGenInference
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain import hub
from dotenv import load_dotenv
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


def get_retriever():
    try:
        model_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embedding = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        loader = CSVLoader(file_path='./wine.csv', encoding='utf-8')
        data = loader.load()
        database = PineconeVectorStore.from_documents(data, embedding, index_name='wine-index')
        retriever = database.as_retriever()
        return retriever
    except Exception as e:
        logging.error(f"Error creating retriever: {e}")
        raise


def get_llm():
    llm = HuggingFaceTextGenInference(
        inference_server_url="https://api-inference.huggingface.co/models/google/gemma-2b",
        max_new_tokens=1024,
        top_k=50,
        temperature=0.1,
        repetition_penalty=1.03,
        server_kwargs={
            "headers": {
                "Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}",
                "Content-Type": "application/json"
            }
        }
    )
    return llm


def get_rag_chain():
    prompt = hub.pull("rlm/rag-prompt")
    llm = get_llm()
    retriever = get_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    conversational_chain = RunnableWithMessageHistory(
        qa_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    ).pick('answer')

    return conversational_chain


def get_ai_message(user_message):
    load_dotenv(verbose=True)
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    rag_chain = get_rag_chain()
    ai_message = rag_chain.stream(
        {
            "input": user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        })
    return ai_message
