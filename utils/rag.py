import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from models.embeddings import get_embedding_model
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS


def process_pdf(uploaded_file):
    try:
        # Stage 1: Save uploaded file to disk temporarily 
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Load the PDF pages 
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # Stage 2: Split into chunks 
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        chunks = splitter.split_documents(pages)

        # Stage 3: Embed and store 
        embeddings = get_embedding_model()
        vector_store = FAISS.from_documents(chunks, embeddings)

        # Clean up the temp file
        os.unlink(tmp_path)

        return vector_store, len(chunks)

    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {str(e)}")


def retrieve_relevant_chunks(vector_store, query: str) -> str:
    try:
        docs = vector_store.similarity_search(query, k=TOP_K_RESULTS)

        if not docs:
            return "No relevant content found in the textbook for this question."

        # Format chunks with their source page for transparency
        context_parts = []
        for i, doc in enumerate(docs, 1):
            page_num = doc.metadata.get("page", "unknown")
            context_parts.append(
                f"[Excerpt {i} — Page {page_num}]\n{doc.page_content}"
            )

        return "\n\n".join(context_parts)

    except Exception as e:
        return f"Error retrieving context: {str(e)}"