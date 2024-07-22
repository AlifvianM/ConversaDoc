import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import warnings

warnings.filterwarnings("ignore")

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

def load_data(dataset_name = 'imdb_top_1000.csv'):
    # movies = pd.read_csv('title.basics.tsv', sep='\t')
    movies = pd.read_csv(dataset_name)
    movies = movies.rename(columns={
        "Overview":"movies_description",
        "Series_Title":"movies_title"
    })
    movies['page_content'] = "Title: " + movies['movies_title'] + "\n" + \
                            "Genre: " + movies['Genre'] + "\n" + \
                            "Description: " + movies['movies_description']
    movies = movies[["page_content", "Poster_Link"]]
    
    docs = DataFrameLoader(movies,page_content_column = "page_content").load()

    # loader = WebBaseLoader(
    #     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    #     bs_kwargs=dict(
    #         parse_only=bs4.SoupStrainer(
    #             class_=("post-content", "post-title", "post-header")
    #         )
    #     ),
    # )
    # docs = loader.load()

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # splits = text_splitter.split_documents(docs)
    return docs

def embbedings_and_store(docs):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model='text-embedding-3-small')
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja", 
    #     model_kwargs={'device': 'cpu'}
    # )
    persist_directory = "chroma_db"
    
    if os.path.exists('chroma_db/chroma.sqlite3'):
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        db = Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory=persist_directory
        )
    # db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    return db

def create_conversational_chain(vector_store):
    # Create llm
    
    retriever = vector_store.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm, 
    #     # chain_type='stuff',
    #     retriever=vector_store.as_retriever(
    #         search_kwargs={"k": 2}
    #     ),
    #     memory=memory,
    #     combine_docs_chain_kwargs={
    #         "prompt":base_prompt
    #     }
    # )

    store = {}


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain

def conversation_chat(query, chain, history=None):
    result = chain.invoke(
        {"input": query},
        config={
            "configurable": {"session_id": "abc123"}
        }
    )["answer"]

    # result = chain.invoke(
    #     {
    #         "question": query,
    #     },
    #     config = {
    #         "configurable": {"session_id": "abc123"}
    #     }
    # )
    # history.append((query, result["answer"]))

    return result

def run_chat(query_input, history = None):
    
    data_loader = load_data('dataset/imdb_top_1000.csv')
    vectorstores = embbedings_and_store(data_loader)
    chain = create_conversational_chain(vectorstores)
    result = conversation_chat(
        query_input, 
        chain=chain, 
        history=history
    )

    return result