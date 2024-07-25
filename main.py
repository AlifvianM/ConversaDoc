import streamlit as st
from streamlit_chat import message

from BE.main import run_chat, load_data

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    
def main(file):
    initialize_session_state()
    if prompt := st.chat_input():

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        response = run_chat(prompt,file=file, history=st.session_state['history'])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

        

if __name__ == '__main__':
    st.header("Give Me Your Document And Ask Anything About It !")
    uploaded_files = st.sidebar.file_uploader("Upload your file here (.pdf)", type=["pdf"], accept_multiple_files=True)
    vectorstores = load_data(uploaded_files)
    main(vectorstores)