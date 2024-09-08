from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

file_path = 'D:\\owd1\\Documents\\GitHub-REPO\\Langchain-Course\\Udemy Course\\PDF\\myCV .pdf'
loader = PyPDFLoader(file_path)

docs = loader.load()

api_key = "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"
os.environ["OPENAI_API_KEY"] = api_key

# Init LLM
llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

results = rag_chain.invoke({"input": "does he have programming skill?"})

print(results)
#print(results["context"][0].page_content)