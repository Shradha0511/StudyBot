#  StudyBot: AI-Powered Textbook Tutor

A RAG-powered chatbot that reads your textbook PDF and answers student questions grounded in the actual course material. Built as part of the NeoStats AI Engineer Case Study.

Built with LangChain, Groq (LLaMA 3.1), HuggingFace Embeddings, FAISS, and Streamlit.

---

## Features

- **Textbook PDF upload** - upload any textbook chapter and have it indexed instantly
- **RAG-powered Q&A** - answers are grounded in your actual textbook, not hallucinated
- **Source page citations** - every answer references the page it came from
- **Concise / Detailed modes** - toggle between short answers and full step-by-step explanations
- **Web search augmentation** - optionally supplement textbook answers with live web results via Tavily
- **Conversation memory** - the bot remembers context across multiple questions in a session

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq — LLaMA 3.1 8B Instant |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (runs locally) |
| Vector Database | FAISS (in-memory, no server needed) |
| RAG Framework | LangChain |
| Web Search | Tavily API |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
AI_UseCase/
├── app.py                  # Streamlit UI
├── requirements.txt
├── README.md
├── .gitignore
├── config/
│   └── config.py           # API keys and settings (gitignored locally)
├── models/
│   ├── llm.py              # Groq LLaMA 3.1 wrapper
│   └── embeddings.py       # HuggingFace MiniLM embeddings
└── utils/
    ├── rag.py              # PDF loading, chunking, FAISS indexing and retrieval
    └── search.py           # Tavily web search
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/Shradha0511/StudyBot.git
cd AI_UseCase
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
.\venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run will download the HuggingFace embedding model (~80MB). This is cached after the first download.

### 4. Get API keys

**Groq** (free):
- Go to [console.groq.com/keys](https://console.groq.com/keys)
- Create a new API key

**Tavily** (free, optional — only needed for web search):
- Go to [app.tavily.com](https://app.tavily.com)
- Create a new API key

### 5. Set environment variables

```bash
# Mac/Linux
export GROQ_API_KEY="your_key_here"
export TAVILY_API_KEY="your_key_here"

# Windows PowerShell
$env:GROQ_API_KEY = "your_key_here"
$env:TAVILY_API_KEY = "your_key_here"
```

### 6. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How to Use

1. Upload a textbook PDF using the sidebar
2. Wait for indexing to complete (~30 seconds on first run)
3. Type your question in the chat input
4. Use **Detailed** mode for concept explanations, **Concise** for quick fact checks
5. Enable **Web Search** to pull in additional context beyond the textbook

---

## How RAG Works

```
Upload PDF
    → saved to temp file on disk
    → loaded page by page via PyPDFLoader
    → split into overlapping 500-character chunks
    → each chunk embedded into a vector by HuggingFace MiniLM (runs locally)
    → all vectors stored in FAISS (in memory)

Ask a question
    → question embedded into a vector using the same model
    → FAISS finds the top 4 most similar chunks
    → chunks + question + chat history sent to Groq LLaMA 3.1
    → answer is grounded in your actual textbook content
```

---



## Notes

- The FAISS index lives in memory and resets when the app restarts — upload your PDF again after a restart
- The HuggingFace model downloads once and is cached locally
- Upload one chapter at a time for faster, more focused answers
