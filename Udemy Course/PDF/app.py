from langchain_community.document_loaders import PyMuPDFLoader  # Import PyMuPDFLoader for PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Corrected import for text splitter
from langchain_openai import OpenAIEmbeddings  # Updated import for OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import fitz  # PyMuPDF library for PDF reading
from langchain.docstore.document import Document  # Import for creating document objects

load_dotenv()

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Improved text splitter using paragraphs and sentences
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],  # Splits by paragraphs, sentences, and words
    chunk_size=500,  # Larger chunks to retain more context
    chunk_overlap=50  # Overlap to maintain continuity between chunks
)

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# Specify your PDF file path
pdf_path = 'D:\\owd1\\Documents\\GitHub-REPO\\Langchain-Course\\Udemy Course\\PDF\\CCAI-413 Lab1.pdf'
pdf_text = extract_text_from_pdf(pdf_path)

# Split the text into chunks
chunks = text_splitter.split_text(pdf_text)

# Convert the text chunks into Document objects
docs = [Document(page_content=chunk) for chunk in chunks]

# Convert the document chunks into embeddings
db = Chroma.from_documents(
    docs,
    embedding=embeddings,
    persist_directory="emb"
)

# Perform a similarity search
results = db.similarity_search(
    "in very short answer, what is NLTK?"
)

# Use the first result and summarize it to ensure a shorter answer
if results:
    short_answer = results[0].page_content.split("\n")[0]  # Get only the first line or sentence
    print(short_answer)
else:
    print("No results found.")