from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Check if OpenAI API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OpenAI API key is not set. Please ensure it is correctly loaded from the environment.")
else:
    print("OpenAI API key is loaded correctly.")

# Initialize embeddings
try:
    embeddings = OpenAIEmbeddings()
    print("OpenAIEmbeddings initialized successfully.")
except Exception as e:
    print(f"Error initializing OpenAIEmbeddings: {e}")

# Check if the file can be read properly
file_path = 'D:\\owd1\\Documents\\GitHub-REPO\\Langchain-Course\\Udemy Course\\facts.txt'

try:
    with open(file_path, 'r') as file:
        content = file.read()
        print("File content preview (first 500 characters):")
        print(content[:500])  # Show the first 500 characters to verify the file content
except FileNotFoundError:
    print(f"Error: File not found at {file_path}. Check the file path.")
except Exception as e:
    print(f"Error reading file: {e}")

# Attempt to load and split the documents
loader = TextLoader(file_path)
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
)

try:
    docs = loader.load_and_split(text_splitter=text_splitter)
    if not docs:
        print("No documents were loaded. The splitting process might be incorrect.")
    else:
        print(f"Loaded {len(docs)} documents. Here are the first few previews:")
        for doc in docs[:5]:  # Print the first 5 document chunks
            print(doc.page_content)
except Exception as e:
    print(f"Error loading and splitting documents: {e}")

# Test embedding generation with a simple query
try:
    test_text = "This is a test."
    test_embedding = embeddings.embed_query(test_text)
    print("Test embedding generated successfully for simple text.")
except Exception as e:
    print(f"Error generating test embedding: {e}")

# Check embedding generation for a loaded document sample
if docs:
    try:
        sample_doc = docs[0].page_content
        sample_embedding = embeddings.embed_query(sample_doc)
        print("Sample document embedding generated successfully.")
    except Exception as e:
        print(f"Error generating embedding for the sample document: {e}")
else:
    print("No documents available to test embedding generation.")

# Create and inspect Chroma database
try:
    # Create Chroma database with the documents
    persist_directory = "emb"
    if not os.path.exists(persist_directory):
        print(f"Persist directory '{persist_directory}' does not exist. Creating it.")
        os.makedirs(persist_directory)
    else:
        print(f"Persist directory '{persist_directory}' is accessible.")

    db = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Chroma database initialized successfully.")
    print(f"Database contains {len(docs)} documents.")
except Exception as e:
    print(f"Error initializing Chroma database: {e}")

# Verify if documents are correctly indexed
try:
    num_indexed_docs = len(db.get_all_documents())
    print(f"Number of documents indexed in Chroma: {num_indexed_docs}")
except Exception as e:
    print(f"Error checking indexed documents: {e}")
# Attempt a broader search query
try:
    broader_query = "facts about the English language"
    results = db.similarity_search(broader_query)
    print(f"Number of results found for the query '{broader_query}': {len(results)}")
    for result in results:
        print("\nResult preview:")
        print(result.page_content)
except Exception as e:
    print(f"Error during similarity search with broader query: {e}")
# Generate an embedding for the search query and inspect it
try:
    search_query = "facts about English language"
    search_embedding = embeddings.embed_query(search_query)
    print("Search query embedding generated successfully.")
    # Optional: Print the first few values of the embedding for inspection
    print(f"First few values of the search embedding: {search_embedding[:5]}")
except Exception as e:
    print(f"Error generating embedding for search query: {e}")

# Attempt a similarity search with adjusted parameters
try:
    results = db.similarity_search("interesting fact", k=10)  # Increase 'k' to allow more results
    print(f"Number of results found with increased 'k': {len(results)}")
    for result in results:
        print("\nResult preview:")
        print(result.page_content)
except Exception as e:
    print(f"Error during similarity search with adjusted parameters: {e}")

# Attempt a simple similarity search
try:
    results = db.similarity_search("interesting fact")
    print(f"Number of results found: {len(results)}")
    for result in results:
        print("\nResult preview:")
        print(result.page_content)
except Exception as e:
    print(f"Error during similarity search: {e}")
