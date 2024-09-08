from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

file_path = 'D:\\owd1\\Documents\\GitHub-REPO\\Langchain-Course\\Udemy Course\\PDF\\the-state-of-ai-in-gcc-countries-and-how-to-overcome-adoption-challenges_final.pdf'
loader = PyPDFLoader(file_path)
docs = loader.load()

api_key = "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"
os.environ["OPENAI_API_KEY"] = api_key

# Init llm
llm = ChatOpenAI(
    openai_api_key=api_key,
    model="gpt-4o-mini")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# Create a vector store using the document chunks and embeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Convert the vector store to a retriever
retriever = vectorstore.as_retriever()

system_prompt = (
    "You are an assistant for answering questions based on the retrieved context. "
    "Use the information provided to generate a direct answer to the question. "
    "Be concise, factual, and accurate. If you don't know the answer based on the context, say that you don't know."
    "\n\nContext:\n{context}"
    "\n\nQuestion: {input}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Combine the document retrieval and question-answering process into a single chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Ask a question and get a direct answer
results = rag_chain.invoke({"input": "summries the main statistics"})

# Print the synthesized answer to the question
print(results["answer"])
