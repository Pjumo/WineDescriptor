from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain import hub
import os
from dotenv import load_dotenv
from langchain_google_vertexai import GemmaLocalKaggle
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
    model_path = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_kwargs = {'device': 'gpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    loader = CSVLoader(file_path='./wine-raitngs.csv', encoding='utf-8')
    data = loader.load()
    database = PineconeVectorStore.from_documents(data, embedding, index_name='wine-index')
    return database.as_retriever()


def get_llm(model_name):
    keras_backend: str = "tensorflow"
    llm = GemmaLocalKaggle(
        model_name=model_name,
        keras_backend=keras_backend,
        max_tokens=1024,
    )
    return llm


def get_qa_chain():
    prompt = hub.pull("rlm/rag-prompt")
    llm = get_llm("gemma_2b_en")
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
    os.environ["KAGGLE_USERNAME"] = os.getenv('KAGGLE_USERNAME')
    os.environ["KAGGLE_KEY"] = os.getenv('KAGGLE_KEY')
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
    rag_chain = get_qa_chain()
    ai_message = rag_chain.stream(
        {
            "input": user_message
        },
        config={
            "configurable": {"session_id": "abc123"}
        })
    return ai_message
