import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import config


def load_pdf_content(pdf_directory: str):
    """
    Load PDF from the particular directory and return the content of the document.
    """
    # 1. Load PDF files from the directory
    if not os.path.exists(pdf_directory):
        print(f"Error: The directory '{pdf_directory}' does not exist.")
        print("Please create the Directory and add the PDF files in it.")
        return []

    all_pdf_docs = []

    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            print(f"Loading PDF: {filepath}")
            try:
                loader = PyPDFLoader(filepath)
                pages = loader.load()
                all_pdf_docs.extend(pages)
            except Exception as e:
                print(f"Error Loading {filepath}: {e}")

    if not all_pdf_docs:
        print("No documents found or No PDF files uploaded in the directory")
    else:
        print(f"Loaded{len(all_pdf_docs)} pages from the PDF documents.")

    return all_pdf_docs


def ingest_all_documents(pdf_directory: str = None, persist_directory: str = None):
    """
    Ingest all the documents from the specific directory and store the embeddings in Chroma vector store.

    The Input PDF are split into chunks, generates embeddings for each chunk, and stores in vector store.
    """
    pdf_directory = pdf_directory or config.PDF_SOURCE_DIRECTORY
    persist_directory = persist_directory or config.CHROMA_PERSIST_DIRECTORY

    print("\n Starting Overall Ingestion Process...")

    # 2. Load PDF content from the directory

    pdf_docs = load_pdf_content(pdf_directory)

    if not pdf_docs:
        print("No document or PDF are loaded. Please check the directory and Upload the files.")
        return

    # 3. Document Splitting into chunks

    chunk_size = config.CHUNK_SIZE
    chunk_overlap = config.CHUNK_OVERLAP

    print(f"\n Splitting documents into chunks with chunk size: {chunk_size} and chunk overlap: {chunk_overlap}...")
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunked_docs = r_splitter.split_documents(pdf_docs)
    print(f"\n Split document into {len(chunked_docs)} chunks.")

    # 4. Embedding Generation and Storing in Vector Store

    model_name = config.EMBEDDING_MODEL_NAME
    print(f"\n Initializing Embedding with model: {model_name}...")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 5. Create Vector store and persist the Embeddings

    import chromadb
    import shutil

    print(f"\n Creating vector store persisting Chroma DB to {persist_directory}...")

    if os.path.exists(persist_directory):
        print(f" Removing existing DB at '{persist_directory}' to prevent HNSW index corruption...")
        try:
            shutil.rmtree(persist_directory)
        except Exception as e:
            print(f" Warning clearing directory: {e}")

    os.makedirs(persist_directory, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_directory)

    vector_db = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        client=client,
        collection_name=getattr(config, "COLLECTION_NAME", "skincare")
    )
    print(
        f"\n Successfully processed {len(chunked_docs)} document chunks and stored the embeddings in chroma vector store at {persist_directory}")
    print(f"\n You can now run your application using the data in {persist_directory} directory.")
    print("\n ---------------------Document Ingestion Process Completed---------------------")


if __name__ == "__main__":
    ingest_all_documents(
        pdf_directory=config.PDF_SOURCE_DIRECTORY,
        persist_directory=config.CHROMA_PERSIST_DIRECTORY
    )