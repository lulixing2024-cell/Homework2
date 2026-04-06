import streamlit as st
import os
import time
import json
import tempfile

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============ API Keys (fill in your keys here) ============
GOOGLE_API_KEY = ""
OPENAI_API_KEY = ""
ANTHROPIC_API_KEY = ""
# Strip whitespace to avoid copy-paste issues
GOOGLE_API_KEY = GOOGLE_API_KEY.strip()
OPENAI_API_KEY = OPENAI_API_KEY.strip()
ANTHROPIC_API_KEY = ANTHROPIC_API_KEY.strip()

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

# ============ Page Config ============
st.set_page_config(page_title="📚 Multi-Agent RAG Chatbot", page_icon="📚", layout="centered")
st.title("📚 Multi-Agent RAG Chatbot")


# ============ Model Factory ============
def get_llm(provider: str, model_name: str, temperature: float = 0.1):
    """Create LLM instance based on selected provider."""
    if provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    elif provider == "OpenAI (GPT)":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif provider == "Anthropic (Claude)":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_embeddings(embedding_provider: str):
    """Create embeddings instance based on selected provider."""
    if embedding_provider == "Gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    elif embedding_provider == "OpenAI (GPT)":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        raise ValueError(f"Unknown embedding provider: {embedding_provider}")


# ============ Model Config ============
MODEL_OPTIONS = {
    "Gemini": [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
    ],
    "OpenAI (GPT)": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "o3-mini",
    ],
    "Anthropic (Claude)": [
        "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20250414",
        "claude-3-5-sonnet-20241022",
    ],
}

# Providers that support embeddings (Claude doesn't have its own embedding API)
EMBEDDING_PROVIDERS = ["Gemini", "OpenAI (GPT)"]


# ============ Sidebar: Model Selection ============
with st.sidebar:
    st.header("🤖 Model Settings")

    # --- LLM Provider & Model ---
    llm_provider = st.selectbox(
        "LLM Provider",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        help="Select which LLM to use for reranking, generation, and verification.",
    )
    llm_model = st.selectbox(
        "Model",
        options=MODEL_OPTIONS[llm_provider],
        index=0,
    )

    # --- Embedding Provider ---
    embedding_provider = st.selectbox(
        "Embedding Provider",
        options=EMBEDDING_PROVIDERS,
        index=0,
        help="Used for building FAISS vector store. Claude has no embedding model, so choose Gemini or OpenAI.",
    )

    st.divider()


# ============ Helper Functions ============
@st.cache_resource(show_spinner="Parsing PDFs and building vector store...")
def build_vectorstore(uploaded_files_data: list[tuple[str, bytes]], embedding_provider: str) -> FAISS:
    """Parse PDFs, split text, build FAISS vector store."""
    all_docs = []

    for file_name, file_bytes in uploaded_files_data:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = file_name
        all_docs.extend(docs)

        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)

    embeddings = get_embeddings(embedding_provider)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


def format_chat_history(messages: list[dict]) -> str:
    """Format recent chat history into a string for context."""
    history = ""
    for msg in messages[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    return history


# ======================================================================
#  FOUR-AGENT PIPELINE
# ======================================================================

def agent0_file_router(
    question: str,
    file_names: list[str],
    llm_provider: str,
    llm_model: str,
) -> list:
    """
    Agent 0 – File Router
    Use LLM to judge which files are relevant to the question.
    """
    llm = get_llm(llm_provider, llm_model, temperature=0.1)

    file_list_str = "\n".join([f"- {fname}" for fname in file_names])
    select_prompt = f"""You are a file relevance judge. Given a user question and a list of PDF file names,
determine which files are likely to contain relevant information.

Return ONLY a JSON array of the relevant file names. No explanation, no markdown fences, just the JSON array.
If all files seem relevant, return all of them.

## User Question
{question}

## Available Files
{file_list_str}
"""

    resp = llm.invoke(select_prompt)
    raw_text = resp.content.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        selected_files = json.loads(raw_text)
        selected_files = [f for f in selected_files if f in file_names]
        if not selected_files:
            selected_files = file_names
    except json.JSONDecodeError:
        selected_files = file_names

    return selected_files


def agent1_broad_retrieval(
    question: str, vectorstore: FAISS, selected_files: list[str], per_file_k: int = 3
) -> list:
    """
    Agent 1 – Broad Retrieval Agent
    Retrieve top `per_file_k` chunks from each selected file.
    """
    all_docs = []
    for fname in selected_files:
        docs = vectorstore.similarity_search(
            question, k=per_file_k, filter={"source": fname}
        )
        all_docs.extend(docs)
    return all_docs


def agent2_rerank_and_generate(
    question: str,
    candidate_docs: list,
    chat_history: list[dict],
    llm_provider: str,
    llm_model: str,
) -> dict:
    """
    Agent 2 – Rerank & Generate Agent
    """
    llm = get_llm(llm_provider, llm_model, temperature=0.1)

    # --- Step A: Rerank ---------------------------------------------------
    numbered_chunks = ""
    for i, doc in enumerate(candidate_docs):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        numbered_chunks += (
            f"[Chunk {i}] (source: {src}, page: {page})\n"
            f"{doc.page_content[:500]}\n\n"
        )

    rerank_prompt = f"""You are a relevance judge. Given a user question and a set of
document chunks, rate each chunk's relevance to the question on a 0-10 scale.

Return ONLY a JSON array of objects, each with "chunk_index" (int) and "score" (int).
No explanation, no markdown fences, just the JSON array.

## Question
{question}

## Candidate Chunks
{numbered_chunks}
"""

    rerank_resp = llm.invoke(rerank_prompt)
    raw_text = rerank_resp.content.strip()
    # Strip possible ```json fences from response
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        scores = json.loads(raw_text)
    except json.JSONDecodeError:
        # Fallback: use original order
        scores = [{"chunk_index": i, "score": 10 - i} for i in range(len(candidate_docs))]

    # Sort by score descending, take all chunks with score >= 5
    scores.sort(key=lambda x: x.get("score", 0), reverse=True)
    qualified = [s for s in scores if s.get("score", 0) >= 5]
    # Fallback: if none qualify, take the top 1
    if not qualified:
        qualified = scores[:1]
    top_indices = [s["chunk_index"] for s in qualified]
    top_docs = [candidate_docs[i] for i in top_indices if i < len(candidate_docs)]
    rerank_detail = scores

    # --- Step B: Generate -------------------------------------------------
    gen_llm = get_llm(llm_provider, llm_model, temperature=0.4)
    context = "\n\n---\n\n".join([doc.page_content for doc in top_docs])
    history_str = format_chat_history(chat_history)

    gen_prompt = f"""You are a knowledgeable document-based Q&A assistant. Answer the user's question
using the provided document excerpts as the primary source. When the documents contain
relevant information, cite specifics from them. When the documents do not fully cover
the question, you may supplement with your own knowledge — but clearly distinguish
between what comes from the documents and what is your general knowledge.

Provide thorough, insightful, and well-structured answers. Do not refuse to answer
just because the documents are incomplete — use them as a foundation and add value.

## Document Excerpts
{context}

## Chat History
{history_str}

## Current Question
{question}

If the user's question is in Chinese, answer in Chinese, otherwise English:"""

    gen_resp = gen_llm.invoke(gen_prompt)

    return {
        "answer": gen_resp.content,
        "top_docs": top_docs,
        "rerank_scores": rerank_detail,
        "top_indices": top_indices,
    }


def agent3_verify_and_format(
    question: str,
    draft_answer: str,
    top_docs: list,
    llm_provider: str,
    llm_model: str,
) -> dict:
    """
    Agent 3 – Verify & Format Agent
    """
    llm = get_llm(llm_provider, llm_model, temperature=0.3)

    context = "\n\n---\n\n".join([doc.page_content for doc in top_docs])

    verify_prompt = f"""You are a quality assurance and formatting agent.

Your tasks:
1. **Fact-check**: Compare the draft answer against the source documents below.
   - If the draft contains claims that directly contradict the documents, correct them.
   - If important information from the documents is missing, add it.
   - Claims based on the model's general knowledge are acceptable as long as they
     don't contradict the documents. Mark them as supplementary if needed.
2. **Enhance**: If the answer is too thin or unhelpful, enrich it with relevant
   analysis, context, or insights that would be valuable to the user.
3. **Format**: Produce a clean, well-structured final answer and put the quick answer at the front.
   - Use bullet points or numbered lists where appropriate.
   - Keep it thorough but well-organized.
   - If the user's question is in Chinese, answer in Chinese, otherwise English:

4. **Verdict**: At the very end, on a new line, output one of:
   ✅ VERIFIED — if the draft was factually consistent
   ⚠️ REVISED — if you had to make corrections

## Source Documents
{context}

## User Question
{question}

## Draft Answer (from previous agent)
{draft_answer}

Output the final polished answer now:"""

    resp = llm.invoke(verify_prompt)
    final_text = resp.content

    # Parse verdict
    verdict = "✅ VERIFIED"
    for line in final_text.strip().splitlines()[::-1]:
        if "VERIFIED" in line:
            verdict = "✅ VERIFIED"
            break
        elif "REVISED" in line:
            verdict = "⚠️ REVISED"
            break

    return {
        "final_answer": final_text,
        "verdict": verdict,
    }


# ======================================================================
#  Orchestrator
# ======================================================================

def multi_agent_rag(
    question: str,
    vectorstore: FAISS,
    chat_history: list[dict],
    file_names: list[str],
    llm_provider: str,
    llm_model: str,
    status_container=None,
) -> dict:
    """Run the full 4-agent pipeline with live status updates."""

    model_label = f"{llm_provider} / {llm_model}"

    # ---- Agent 0 ----
    if status_container:
        status_container.update(
            label=f"📂 Agent 0: Routing — selecting relevant files from {len(file_names)}...",
            state="running",
        )
    selected_files = agent0_file_router(
        question, file_names,
        llm_provider=llm_provider, llm_model=llm_model,
    )

    # ---- Agent 1 ----
    if status_container:
        status_container.update(
            label=f"🔍 Agent 1: Retrieving 3 chunks × {len(selected_files)} selected files...",
            state="running",
        )
    candidate_docs = agent1_broad_retrieval(question, vectorstore, selected_files)

    # ---- Agent 2 ----
    if status_container:
        status_container.update(
            label=f"🧠 Agent 2: Reranking & Generating ({model_label})...",
            state="running",
        )
    agent2_result = agent2_rerank_and_generate(
        question, candidate_docs, chat_history,
        llm_provider=llm_provider, llm_model=llm_model,
    )

    # ---- Agent 3 ----
    if status_container:
        status_container.update(
            label=f"✅ Agent 3: Verifying & Formatting ({model_label})...",
            state="running",
        )
    agent3_result = agent3_verify_and_format(
        question, agent2_result["answer"], agent2_result["top_docs"],
        llm_provider=llm_provider, llm_model=llm_model,
    )

    if status_container:
        status_container.update(label="✨ Pipeline complete!", state="complete")

    return {
        "selected_files": selected_files,
        "candidate_docs": candidate_docs,
        "rerank_scores": agent2_result["rerank_scores"],
        "top_indices": agent2_result["top_indices"],
        "top_docs": agent2_result["top_docs"],
        "draft_answer": agent2_result["answer"],
        "final_answer": agent3_result["final_answer"],
        "verdict": agent3_result["verdict"],
        "model_used": model_label,
    }


# ============ Sidebar: PDF Upload ============
with st.sidebar:
    st.header("📄 Upload PDF Files")
    uploaded_files = st.file_uploader(
        "Browse PDF files (multi-choice accepted)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} files uploaded")
        for f in uploaded_files:
            st.write(f"- {f.name} ({f.size / 1024:.1f} KB)")

    st.divider()
    if st.button("🗑️ Clear Vector Store Cache", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    st.divider()
    st.markdown(
        """
        **How it works — 4-Agent Pipeline**

        | Agent | Role |
        |-------|------|
        | 📂 Agent 0 | File Router — select relevant files |
        | 🔍 Agent 1 | Broad Retrieval — 3 chunks per file |
        | 🧠 Agent 2 | Rerank → Top 3 → Generate |
        | ✅ Agent 3 | Fact-check & format |
        """
    )

# ============ Build Vector Store ============
if uploaded_files:
    files_data = [(f.name, f.read()) for f in uploaded_files]
    file_names = [f.name for f in uploaded_files]
    for f in uploaded_files:
        f.seek(0)
    vectorstore = build_vectorstore(files_data, embedding_provider)
    vectorstore_ready = True
else:
    vectorstore_ready = False
    file_names = []

# ============ Current Model Badge ============
st.caption(f"🤖 Current model: **{llm_provider} / {llm_model}** ｜ Embeddings: **{embedding_provider}**")

# ============ Chat Interface ============
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("agent_details"):
            with st.expander("🔗 Agent Pipeline Details"):
                st.markdown(message["agent_details"])

# Chat input
if not vectorstore_ready:
    st.info("👈 Please upload PDF files in the sidebar first")

user_input = st.chat_input(
    "Ask a question about your documents...",
    disabled=not vectorstore_ready,
)

if user_input and vectorstore_ready:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run multi-agent pipeline
    with st.chat_message("assistant"):
        with st.status("Running 4-Agent Pipeline...", expanded=True) as status:
            result = multi_agent_rag(
                user_input, vectorstore, st.session_state.messages, file_names,
                llm_provider=llm_provider, llm_model=llm_model,
                status_container=status,
            )

        # Simulated streaming of final answer
        message_placeholder = st.empty()
        full_response = ""
        display_answer = result["final_answer"].replace("$", r"\$")
        for chunk in display_answer.split():
        
            full_response += chunk + " "
            message_placeholder.markdown(full_response)
            time.sleep(0.01)

        # ---- Build agent pipeline details for expander ----
        details_md = ""
        details_md += f"**Model Used**: `{result['model_used']}`\n\n"

        # Agent 0 summary
        details_md += "### 📂 Agent 0 — File Router\n"
        details_md += f"Selected **{len(result['selected_files'])}** of {len(file_names)} files: **{', '.join(result['selected_files'])}**\n\n"

        # Agent 1 summary
        details_md += "### 🔍 Agent 1 — Broad Retrieval\n"
        details_md += f"Retrieved **{len(result['candidate_docs'])}** chunks (3 per selected file).\n\n"
        for i, doc in enumerate(result["candidate_docs"]):
            src = doc.metadata.get("source", "?")
            page = doc.metadata.get("page", "?")
            snippet = doc.page_content[:100].replace("\n", " ")
            details_md += f"- **Chunk {i}** [{src}, p.{page}]: {snippet}…\n"
        details_md += "\n"

        # Agent 2 summary
        details_md += "### 🧠 Agent 2 — Rerank & Generate\n"
        details_md += "Relevance scores:\n\n"
        for s in result["rerank_scores"]:
            idx = s.get("chunk_index", "?")
            score = s.get("score", "?")
            marker = " ⬅️ selected" if idx in result["top_indices"] else ""
            details_md += f"- Chunk {idx}: **{score}/10**{marker}\n"
        details_md += f"\n**{len(result['top_docs'])}** chunks scored ≥ 5 selected for answer generation.\n\n"
        details_md += "**Agent 2 Draft Answer:**\n\n"
        draft_display = result['draft_answer'].replace('$', r'\$')
        details_md += f"{draft_display}\n\n"

        # Agent 3 summary
        details_md += "### ✅ Agent 3 — Verify & Format\n"
        details_md += f"Verdict: **{result['verdict']}**\n"

        with st.expander("🔗 Agent Pipeline Details"):
            st.markdown(details_md)

    # Save to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["final_answer"].replace("$", r"\$"),
        "agent_details": details_md,
    })