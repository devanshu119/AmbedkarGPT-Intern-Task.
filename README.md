# AmbedkarGPT-Intern-Task.
mbedkarGPT is a Retrieval-Augmented Generation (RAG) application built with:

LangChain 1.x

ChromaDB (local vector database)

HuggingFace Embeddings

Ollama (local LLM runner)

Python 3.10+

This system loads a speech text file (speech.txt), splits it into chunks, embeds it, stores vectors locally, and lets users ask questions only from this document, producing accurate answers via RAG.

🚀 Features

Fully local (no external API calls)

Uses Ollama to run models like mistral, llama3, etc.

Automatic chunking + embedding + retrieval

LCEL-based RAG pipeline (compatible with LangChain 1.x)

Interactive CLI question-answer interface

Debug-friendly with detailed logs

📁 Project Structure
project/
│
├── main_debug.py       # Main RAG application
├── speech.txt          # Document used for Q&A
│
├── chroma_db/          # Auto-generated vector database
│
└── README.md

🔧 Requirements
1. Python 3.10 or higher
2. Install Dependencies

Use a virtual environment:

python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows


Install required packages:

pip install langchain-community langchain-core chromadb sentence-transformers

3. Install & Start Ollama

Download Ollama:
https://ollama.com

Pull your model:

ollama pull mistral


Start server automatically (Ollama runs as a background service):

ollama serve

📘 Usage
1. Ensure speech.txt exists in the same directory.

It should contain the text from which answers will be generated.

2. Run the app
First time (rebuild vector DB):
python main_debug.py --rebuild


Normal run:

python main_debug.py

3. Ask Questions

When the CLI starts:

--- Ambedkar Q&A System Ready ---
Ask questions based ONLY on the content of speech.txt.


Example:

Your Question: What is the main message of the speech?


Exit anytime:

Your Question: exit

🧩 How It Works (RAG Pipeline)

Load speech.txt

Split into 500-character overlapping chunks

Embed using sentence-transformers/all-MiniLM-L6-v2

Store in Chroma vector DB

Retrieve relevant chunks at query time

Generate answer using the Ollama model (Mistral)

The retrieval + generation sequence is defined using LangChain’s new LCEL:

Retriever → Prompt → Ollama LLM

🛠️ Configuration

You can modify these variables inside main_debug.py:

DOCUMENT_PATH = "speech.txt"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral"
CHROMA_DB_DIR = "chroma_db"
OLLAMA_BASE_URL = "http://localhost:11434"

🐞 Debug Info

The script prints detailed debug messages:

Whether speech.txt exists

Number of chunks created

Whether embeddings initialized

Whether vector DB loaded or rebuilt

Full exception tracebacks on errors

Raw LLM output for inspection

💡 Troubleshooting
ImportError: RetrievalQA not found

This project uses LangChain 1.x → RetrievalQA is removed.
Solution: We use the LCEL RAG pipeline instead (already implemented).

Ollama connection errors

Ensure Ollama is running:

ollama serve

speech.txt empty or missing

Place your document in the same folder.
