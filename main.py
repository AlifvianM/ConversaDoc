import streamlit as st
from streamlit_chat import message

from BE.main import run_chat, load_data

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # if 'generated' not in st.session_state:
    #     st.session_state['generated'] = ["Hello! Tell me about your taste of movies 🤗"]

    # if 'past' not in st.session_state:
    #     st.session_state['past'] = ["Hey! 👋"]

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    
    # Delete all the items in Session state
    # for key in st.session_state.keys():
    #     del st.session_state[key]

    
def main(file):
    initialize_session_state()
    # reply_container = st.container()
    # container = st.container()

    # with container:
        # user_input = st.chat_input("What you wanna ask ?")

    #     if user_input:
    #         with st.spinner('Generating response...'):
    #             output = run_chat(user_input, st.session_state['history'])
    #             st.session_state['past'].append(user_input)
    #             st.session_state['generated'].append(output)

    # if st.session_state['generated']:
    #     with reply_container:
    #         for i in range(len(st.session_state['generated'])):
    #             message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars", seed="Aneka")
    #             message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed="Aneka")

    if prompt := st.chat_input():

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        response = run_chat(prompt, file, history=st.session_state['history'])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

        

if __name__ == '__main__':
    st.header("Give Me Your Document And Ask Anything About It !")
    uploaded_files = st.sidebar.file_uploader("Upload your file here (.pdf, .docx, or .txt)", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    vectorstores = load_data(uploaded_files)
    main(vectorstores)