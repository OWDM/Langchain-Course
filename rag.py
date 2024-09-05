from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import retrieval_qa
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st

from watsonxlangchain import LangChainInterface


st.title("ask watsonx")

prompt = st.chat_input("Pass Your Prompt here:")

if prompt:
    st.chat_message('user').markdown(prompt)