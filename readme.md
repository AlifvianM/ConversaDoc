# ConversaDoc AI
This project was developed as a basis for practicing LLM and retrieval augemented generation (RAG) built using python with the support of langchain and openai as LLM processing and streamlit for deployment. The project we created is focused on being able to help users in selecting and recommending movies according to what the user wants. Because of this, RAG has an important role in providing context to the model where the context is obtained from user.

<p align="center">
  <img src="image.png" width="85%" height="85%">
</p>

## How it works
The way our app works is by receiving a document from the user which is immediately processed by the program to retrieve the context that will be stored in vectorstores. After that the user will give a prompt to interact with LLM and after that LLM will process until the answer can be returned to the user and so on.

## Installation
* prepare the environment and activate it
* install packages `pip install -r requirements.txt`
* run streamlit server `streamlit run main.py`
<hr>

## Big Thanks To
* [Syahvan Alviansyah](https://www.linkedin.com/in/syahvanalviansyah/)