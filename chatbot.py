import os
import chromadb
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ================================
# 🔹 Ensure ChromaDB Directory Exists
# ================================
CHROMA_DB_PATH = "./chroma_db"
if not os.path.exists(CHROMA_DB_PATH):
    os.makedirs(CHROMA_DB_PATH)

# ================================
# 🔹 Load Ollama DeepSeek Model
# ================================
llm = OllamaLLM(model="deepseek-r1:1.5b")  # ✅ Ensure model is correctly specified

# ================================
# 🔹 Load Embedding Model
# ================================
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ================================
# 🔹 Initialize ChromaDB
# ================================
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection("eye_health")  # ✅ Collection for storing knowledge

# ================================
# 🔹 Load PDFs and Store Embeddings
# ================================
def load_pdfs(pdf_folder="data"):
    """ Loads PDFs, splits text into chunks, and stores embeddings in ChromaDB. Runs only once. """
    if os.path.exists("pdfs_loaded.flag"):
        print("✔ PDFs are already loaded, skipping...")
        return

    print("🔄 Loading PDFs into ChromaDB...")
    
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            try:
                pdf_loader = PyPDFLoader(pdf_path)
                docs = pdf_loader.load()
            except ModuleNotFoundError:
                print("❌ ERROR: Install 'pypdf' → `pip install pypdf`")
                continue
            except Exception as e:
                print(f"❌ Error loading PDF {pdf_file}: {e}")
                continue

            if docs:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_documents(docs)

                for i, chunk in enumerate(chunks):
                    text = chunk.page_content
                    try:
                        embedding = embedding_model.embed_query(text)
                        collection.add(
                            documents=[text],
                            embeddings=[embedding],
                            ids=[f"{pdf_file}_{i}"]
                        )
                    except Exception as e:
                        print(f"❌ Error storing chunk {i} of {pdf_file}: {e}")

    # ✅ Create a flag file after successful processing
    with open("pdfs_loaded.flag", "w") as f:
        f.write("PDFs loaded")
    
    print("✅ PDF Processing & Embedding Done!")

# ================================
# 🔹 Query Knowledge Base Function
# ================================
# 🔹 Query Knowledge Base Function
# ================================
def query_knowledge_base(user_query):
    """
    Retrieves relevant context from ChromaDB and generates a response using DeepSeek-R1-1.5B via Ollama.
    """
    print(f"\n🔍 Query: {user_query}")

    try:
        # Retrieve top matching text from ChromaDB
        results = collection.query(query_embeddings=[embedding_model.embed_query(user_query)], n_results=3)
        
        # Debugging: Print raw ChromaDB results
        print("📌 ChromaDB Results:", results)

        retrieved_docs = results.get("documents", [])
        if not retrieved_docs or not retrieved_docs[0]:
            return "No relevant information found in the knowledge base."

        # Combine retrieved text chunks
        context = "\n".join(retrieved_docs[0])

        # Debugging: Print retrieved context
        print("📌 Retrieved Context:\n", context)

        # Format prompt for DeepSeek-R1-1.5B
        prompt = f"Answer the following based on the given context:\n\nContext: {context}\n\nQuestion: {user_query}\n\nAnswer:"

        # ✅ Generate response using Ollama
        response = llm.invoke(prompt)

        return response
    except Exception as e:
        print(f"❌ Chatbot Backend Error: {e}")
        return "❌ Error: Backend Issue"

# ================================
# 🔹 Example Usage
# ================================
if __name__ == "__main__":
    print("\n🔹 Loading PDFs & Creating Embeddings...")
    load_pdfs()  # Load PDFs only if not already loaded

    user_query = "What are the causes of Cataract?"
    response = query_knowledge_base(user_query)
    print("\nChatbot:", response)
