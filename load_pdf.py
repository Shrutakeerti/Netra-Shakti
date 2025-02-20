# load_data.py
from chatbot import load_pdfs

if __name__ == "__main__":
    print("🔄 Loading PDFs into ChromaDB...")
    load_pdfs()
    print("✅ PDFs successfully loaded into ChromaDB!")
