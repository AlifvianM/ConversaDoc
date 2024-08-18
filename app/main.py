import streamlit as st
from streamlit_chat import message
from PyPDF2 import PdfReader

from BE.main import (
    run_chat, 
    load_data, 
    embbedings_and_store, 
    create_conversational_chain,
    conversation_chat,
    start_conversation
)

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        # st.session_state["generated"] = [{"role": "assistant", "content": "How can I help you?"}]
        st.session_state["generated"] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]
    
    if 'ready' not in st.session_state:
        st.session_state['ready'] = False

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        user_input = st.chat_input("Ask me something....")

        if user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars", seed="Aneka")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed="Aneka")


def main():
    # Initialize session state
    initialize_session_state()
    
    print('Start Main :', st.session_state.ready)
    # load_dotenv()  
    # groq_api_key = os.environ['GROQ_API_KEY']
    st.set_page_config(page_title="Ask your Document")
    st.header("Chat With Your Doc !")
    # linkedin = "https://www.linkedin.com/in/syahvanalviansyah/"
    # st.markdown("a Multi-Documents ChatBot App by [Syahvan Alviansyah](%s) 👨🏻‍💻" % linkedin)
    # Initialize Streamlit
    st.sidebar.title("Upload Your Doc Here")
    uploaded_files = st.sidebar.file_uploader("Upload your file here (.pdf)", type=["pdf"], accept_multiple_files=True)

    # Create embeddings
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
    #                                        model_kwargs={'device': 'cpu'})
    if st.session_state.ready == False:
        if uploaded_files:
            # extract text from uploaded files
            all_text = ""
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    pdf_reader = PdfReader(uploaded_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                else:
                    st.write(f"Unsupported file type: {uploaded_file.name}")
                    continue
                all_text += text
            
            with st.spinner('Analyze Document...'):
                # Create vector store
                # vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                vector_store = embbedings_and_store(all_text)
                st.session_state['vector'] = vector_store
                del vector_store
            
            st.session_state.ready = True
        else:
            st.warning('⚠️ Please upload your document in the sidebar first in order to access the chatbot!')
    if st.session_state.ready == True:
        # Create the chain object
        chain = create_conversational_chain(st.session_state.vector)        
        display_chat_history(chain)

if __name__ == "__main__":
    main()