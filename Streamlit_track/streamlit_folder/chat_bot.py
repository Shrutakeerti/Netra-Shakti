import os
import chromadb
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ================================
# 🔹 Paths & Configurations
# ================================
CHROMA_DB_PATH = "./chroma_db"
PDF_FOLDER = "data"

# Ensure ChromaDB directory exists
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Initialize Ollama DeepSeek Model
llm = OllamaLLM(model="deepseek-r1:1.5b")

# Load Hugging Face Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection("eye_health")

# ================================
# 🔹 Load PDFs & Create Embeddings (Runs Only If Necessary)
# ================================
def load_pdfs():
    """ Loads PDFs from `data/` folder, splits text, and stores embeddings in ChromaDB. """
    
    # Check if collection already contains data
    if collection.count() > 0:
        print("✅ Embeddings already exist in ChromaDB. Skipping PDF processing.")
        return

    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    
    if len(pdf_files) < 2:
        print("❌ ERROR: At least 2 PDFs required in 'data/' folder.")
        return

    print("🔄 Processing PDFs...")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        pdf_loader = PyPDFLoader(pdf_path)
        docs = pdf_loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        # Store embeddings in ChromaDB
        for i, chunk in enumerate(chunks):
            text = chunk.page_content
            embedding = embedding_model.embed_query(text)
            collection.add(documents=[text], embeddings=[embedding], ids=[f"{pdf_file}_{i}"])

    print("✅ PDFs Processed & Embeddings Created!")

# ================================
# 🔹 Query Knowledge Base
# ================================
def query_knowledge_base(user_query):
    """ Retrieves relevant data from ChromaDB and generates a response using DeepSeek-R1-1.5B. """
    print(f"\n🔍 User Query: {user_query}")

    # Retrieve top matching text from ChromaDB
    results = collection.query(query_embeddings=[embedding_model.embed_query(user_query)], n_results=3)
    retrieved_docs = results.get("documents", [])

    if not retrieved_docs or not retrieved_docs[0]:
        return "No relevant information found."

    # Combine retrieved text
    context = "\n".join(retrieved_docs[0])

    # Format prompt
    prompt = f"Context: {context}\n\nQuestion: {user_query}\n\nAnswer:"
    
    # Generate response
    response = llm.invoke(prompt)

    return response

# ================================
# 🔹 Run Only When Needed
# ================================
if __name__ == "__main__":
    print("\n🔹 Checking & Loading PDFs (if necessary)...")
    load_pdfs()  # Runs only if embeddings are not already in ChromaDB

    user_query = "What are the causes of Cataract?"
    response = query_knowledge_base(user_query)
    print("\nChatbot Response:", response)
