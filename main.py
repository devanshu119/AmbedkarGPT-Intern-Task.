# main_debug.py
import os
import sys
import argparse
import traceback
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# NEW imports replacing RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


DOCUMENT_PATH = "speech.txt"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral"
CHROMA_DB_DIR = "chroma_db"
OLLAMA_BASE_URL = "http://localhost:11434"  # explicit base_url for Ollama


def check_speech_file(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] {path} not found in current directory: {os.getcwd()}")
        return False
    if p.stat().st_size == 0:
        print(f"[ERROR] {path} exists but is empty (size 0). Please add the speech text.")
        return False
    print(f"[OK] Found {path} (size: {p.stat().st_size} bytes).")
    return True


def build_vector_store(texts, embeddings, persist_dir, rebuild=False):
    persist_path = Path(persist_dir)
    if rebuild and persist_path.exists():
        print(f"[DEBUG] --rebuild requested. Deleting existing directory: {persist_dir}")
        try:
            import shutil
            shutil.rmtree(persist_path)
            print(f"[DEBUG] Deleted {persist_dir}")
        except Exception as e:
            print(f"[WARN] Failed to delete {persist_dir}: {e}")

    print(f"[DEBUG] Creating/Loading Chroma vector store in '{persist_dir}' ...")
    try:
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vectorstore.persist()
        print("[OK] Vector store created/loaded and persisted.")
        return vectorstore
    except Exception:
        print("[ERROR] Failed to create/load Chroma vector store:")
        traceback.print_exc()
        raise


def setup_rag_pipeline(rebuild=False):
    print("[STEP] 1) Verify speech.txt")
    if not check_speech_file(DOCUMENT_PATH):
        return None

    print("[STEP] 2) Load document using TextLoader")
    try:
        loader = TextLoader(DOCUMENT_PATH)
        documents = loader.load()
        print(f"[DEBUG] TextLoader returned {len(documents)} documents")
    except Exception:
        print("[ERROR] Exception while loading document:")
        traceback.print_exc()
        return None

    print("[STEP] 3) Split into chunks")
    try:
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        print(f"[DEBUG] Created {len(texts)} chunks.")
        if len(texts) == 0:
            print("[ERROR] No chunks created. Check speech.txt.")
            return None
    except Exception:
        print("[ERROR] Exception during splitting:")
        traceback.print_exc()
        return None

    print("[STEP] 4) Initialize embeddings")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        print("[OK] Embeddings initialized.")
    except Exception:
        print("[ERROR] Failed to initialize embeddings:")
        traceback.print_exc()
        return None

    print("[STEP] 5) Create/Load Chroma vector store")
    try:
        vectorstore = build_vector_store(texts, embeddings, CHROMA_DB_DIR, rebuild=rebuild)
    except Exception:
        print("[ERROR] Could not build vector store.")
        return None

    print("[STEP] 6) Initialize Ollama LLM")
    try:
        llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
        print(f"[DEBUG] Ollama LLM created: {llm}")
    except Exception:
        print("[ERROR] Failed to initialize Ollama.")
        traceback.print_exc()
        return None

    print("[STEP] 7) Build LCEL RAG chain (replacement for RetrievalQA)")
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a helpful assistant. Use ONLY the context to answer.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
        )

        rag_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )

        print("[OK] LCEL RAG chain initialized.")
        return rag_chain

    except Exception:
        print("[ERROR] Failed to initialize LCEL RAG chain:")
        traceback.print_exc()
        return None


def interactive_loop(rag_chain):
    print("\n--- Ambedkar Q&A System Ready ---")
    print("Ask questions based ONLY on speech.txt. Type 'exit' to quit.")
    print("-" * 40)

    while True:
        try:
            question = input("Your Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting.")
            break

        if question.lower() in ("exit", "quit"):
            print("[INFO] Exiting. Goodbye.")
            break
        if not question:
            continue

        try:
            print("[DEBUG] Running RAG pipeline...")
            answer = rag_chain.invoke(question)
            print("\nAnswer:\n", answer)
            print("-" * 40)
        except Exception:
            print("[ERROR] Exception during generation:")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Ambedkar Q&A debug runner")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild chroma_db")
    args = parser.parse_args()

    print("[INFO] Starting setup (cwd: {})".format(os.getcwd()))
    rag_chain = setup_rag_pipeline(rebuild=args.rebuild)
    if rag_chain is None:
        print("[ERROR] Setup failed. Exiting.")
        sys.exit(1)

    interactive_loop(rag_chain)


if __name__ == "__main__":
    main()

