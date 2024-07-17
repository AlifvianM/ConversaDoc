import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')


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
    return docs

def embbedings_and_store(docs):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    persist_directory = "chroma_db"

    # save ke local
    db = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=persist_directory
    )
    return db

def create_conversational_chain(vector_store):
    # Create llm
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

def conversation_chat(query, chain, history=None):
    result = chain.invoke(
        {
            "question": query, 
            # "chat_history": history
        }
    )
    # history.append((query, result["answer"]))
    return result["answer"]

def run_chat(query_input, history = None):
    
    data_loader = load_data('dataset/imdb_top_1000.csv')
    vectorstores = embbedings_and_store(data_loader)
    chain = create_conversational_chain(vectorstores)
    result = conversation_chat(
        query_input, 
        chain=chain, 
        # history=history
    )

    return result