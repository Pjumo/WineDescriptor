from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain import hub
import os
from dotenv import load_dotenv
from langchain_google_vertexai import GemmaLocalKaggle
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders.csv_loader import CSVLoader


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
    return qa_chain


def get_ai_message(user_message):
    load_dotenv(verbose=True)
    os.environ["KAGGLE_USERNAME"] = os.getenv('KAGGLE_USERNAME')
    os.environ["KAGGLE_KEY"] = os.getenv('KAGGLE_KEY')
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
    qa_chain = get_qa_chain()
    ai_message = qa_chain({"query": user_message})
    return ai_message['result']
