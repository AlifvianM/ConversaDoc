import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import bs4
from PyPDF2 import PdfReader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain


import warnings

warnings.filterwarnings("ignore")

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
def load_data(
        uploaded_files,
        dataset_name = 'imdb_top_1000.csv'
    ):
    TEXT = ""
    # loader = WebBaseLoader(
    # web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    # bs_kwargs=dict(
    #         parse_only=bs4.SoupStrainer(
    #             class_=("post-content", "post-title", "post-header")
    #         )
    #     ),
    # )
    # docs = loader.load()

    # for file in uploaded_files:
        # import pdb;pdb.set_trace()
    if uploaded_files.type == "application/pdf":
        pdf_reader = PdfReader()
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    # TEXT += text
    docs = TEXT
    return docs

def embbedings_and_store(docs):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    persist_directory = "chroma_db"
    
    text_splitter = CharacterTextSplitter(separator=" ", chunk_size=5000, chunk_overlap=1000, length_function=len)
    text_chunks = text_splitter.split_text(docs)
    
    vectorstore = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory=persist_directory)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    return retriever

def create_conversational_chain(retriever):
    ### Contextualize question ###
    contextualize_q_system_prompt = (
        """
        Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

        Chat History:

        Human: <CHAT FROM HUMAN>
        Assistant: <CHAT RESPONSE>
        Follow Up Input: <CURRENT CHAT FROM HUMAN>
        Standalone question:
        """
    )
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
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
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

def start_conversation(vector_embeddings):
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_embeddings,
        memory=memory
    )

    return conversation

def conversation_chat(query, chain, history=None):
    result = chain.invoke(
        {"input": query},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )["answer"]
    return result

def run_chat(query_input, file, history = None):
    vectorstores = embbedings_and_store(file)
    chain = create_conversational_chain(vectorstores)
    result = conversation_chat(
        query_input, 
        chain=chain, 
        history=history
    )
    return result