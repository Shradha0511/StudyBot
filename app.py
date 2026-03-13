import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.llm import get_chatgroq_model
from utils.rag import process_pdf, retrieve_relevant_chunks
from utils.search import web_search


# ---------------------------------------------------------------------------
# PROMPT BUILDER
# The key insight: you never send the raw question to the LLM.
# You ENRICH it with textbook context (+ optional web search) first.
# ---------------------------------------------------------------------------

def build_system_prompt(mode: str) -> str:
    """
    Returns the system prompt based on the user's chosen response style.
    
    The system prompt sets the LLM's personality for the entire conversation.
    Changing the mode is literally just changing this string — no new model needed.
    """
    base = (
        "You are StudyBot, a friendly and knowledgeable tutor helping students "
        "understand their textbook material. You explain concepts clearly, use "
        "examples where helpful, and always base your answers on the provided "
        "textbook excerpts. If the answer is not in the textbook, say so honestly."
    )

    if mode == "Concise":
        return base + (
            " Keep your answers short and to the point — 2 to 4 sentences max. "
            "Prioritize clarity over completeness."
        )
    else:  # Detailed
        return base + (
            " Give thorough, detailed explanations. Break down complex ideas step "
            "by step. Use bullet points or numbered lists when they aid clarity."
        )


def build_full_prompt(
    user_question: str,
    textbook_context: str,
    web_context: str,
    chat_history: list,
    mode: str,
) -> list:
    """
    Assembles the full message list to send to the LLM.
    
    Structure:
        SystemMessage  ← sets personality + response style
        HumanMessage   ← past user messages
        AIMessage      ← past bot responses
        HumanMessage   ← current question, enriched with context
    """
    system_prompt = build_system_prompt(mode)
    messages = [SystemMessage(content=system_prompt)]

    # Inject conversation history (so the bot remembers earlier turns)
    for msg in chat_history[:-1]:   # all but the last (current) message
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Build the enriched current question
    enriched_question = f"Student's question: {user_question}\n\n"

    if textbook_context:
        enriched_question += (
            f"--- Relevant Textbook Excerpts ---\n{textbook_context}\n\n"
        )

    if web_context:
        enriched_question += (
            f"--- Additional Web Context ---\n{web_context}\n\n"
        )

    enriched_question += (
        "Please answer the student's question based on the above context."
    )

    messages.append(HumanMessage(content=enriched_question))
    return messages


# ---------------------------------------------------------------------------
# PAGE: CHAT
# ---------------------------------------------------------------------------

def chat_page():
    st.title("StudyBot — Your Textbook Tutor")
    st.caption("Upload your textbook PDF and ask anything about it.")

    # --- Sidebar controls ---
    with st.sidebar:
        st.header("Settings")

        uploaded_file = st.file_uploader(
            "Upload Textbook PDF",
            type=["pdf"],
            help="Upload the PDF of the textbook chapter you're studying.",
        )

        st.divider()

        response_mode = st.radio(
            "Response Style",
            ["Concise", "Detailed"],
            index=0,
            help="Concise = short answers. Detailed = full explanations.",
        )

        use_web_search = st.toggle(
            "Supplement with web search",
            value=False,
            help="When ON, StudyBot will also search the web to enrich answers.",
        )

        st.divider()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- Process PDF when uploaded ---
    # session_state persists across Streamlit reruns (Streamlit re-runs the
    # whole script on every interaction, so we MUST store things in session_state)
    if uploaded_file:
        # Only re-process if a NEW file was uploaded
        if (
            "last_uploaded_file" not in st.session_state
            or st.session_state.last_uploaded_file != uploaded_file.name
        ):
            with st.spinner("Reading and indexing your textbook... (first time takes ~30s)"):
                try:
                    vector_store, num_chunks = process_pdf(uploaded_file)
                    st.session_state.vector_store = vector_store
                    st.session_state.last_uploaded_file = uploaded_file.name
                    st.session_state.messages = []   # fresh chat for new book
                    st.success(f"Indexed {num_chunks} text chunks from **{uploaded_file.name}**. Ask away!")
                except Exception as e:
                    st.error(f"Error processing PDF: {e}")
                    return
    else:
        st.info("Upload a textbook PDF from the sidebar to get started.")
        return

    # --- Initialize chat history ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Display past messages ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat input ---
    if prompt := st.chat_input("Ask a question about your textbook..."):

        # Save and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching textbook..."):

                # Step 1: Retrieve relevant chunks from textbook
                textbook_context = retrieve_relevant_chunks(
                    st.session_state.vector_store, prompt
                )

                # Step 2 (optional): Web search for extra context
                web_context = ""
                if use_web_search:
                    with st.spinner("🌐 Searching the web..."):
                        web_context = web_search(prompt)

                # Step 3: Build enriched prompt and call LLM
                try:
                    chat_model = get_chatgroq_model()
                    full_messages = build_full_prompt(
                        user_question=prompt,
                        textbook_context=textbook_context,
                        web_context=web_context,
                        chat_history=st.session_state.messages,
                        mode=response_mode,
                    )
                    response = chat_model.invoke(full_messages)
                    answer = response.content

                except Exception as e:
                    answer = f"Error getting response: {str(e)}"

                st.markdown(answer)

        # Save bot response to history
        st.session_state.messages.append({"role": "assistant", "content": answer})


# ---------------------------------------------------------------------------
# PAGE: INSTRUCTIONS
# ---------------------------------------------------------------------------

def instructions_page():
    st.title("Enjoy Learning with StudyBot")

    st.markdown("""
## What is StudyBot?
A RAG-powered tutor that reads your textbook PDF and answers questions about it.
It retrieves the most relevant excerpts before answering — so it stays grounded
in your actual course material instead of hallucinating.







---

## How RAG works (plain English)

```
You upload a PDF
    → saved to temp file
    → loaded page by page
    → split into 500-char overlapping chunks
    → each chunk embedded into numbers by HuggingFace model
    → stored in FAISS (local vector database)

You ask a question
    → question embedded into numbers (same model)
    → FAISS finds top 4 most similar chunks
    → chunks + question sent to Groq LLM
    → LLM answers using only your textbook content
```

---

## Tips
- **Upload one chapter at a time** for faster, more focused answers
- **Concise mode** is great for quick fact checks
- **Detailed mode** is great for understanding complex topics
- The first PDF upload takes ~30 seconds (model downloads once, then it's cached)
""")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="StudyBot — Textbook Tutor",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("StudyBot")
        st.caption("Your AI-powered textbook tutor")
        st.divider()
        page = st.radio("Navigate", ["Chat", "Instructions"], index=0)

    if page == "Instructions":
        instructions_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()