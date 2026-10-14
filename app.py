import sys
from pathlib import Path
import streamlit as st

# Add src/personal_rag to sys.path so modules import seamlessly
SRC_DIR = Path(__file__).resolve().parent / "src" / "personal_rag"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    PDF_PATH,
    GITHUB_USERNAME,
    GITHUB_DATA_DIR,
    FAISS_INDEX_PATH,
    DATA_DIR,
    RETRIEVER_K,
)
from pipeline import PersonalRAG
from github_data_ingest import ingest_github

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Personal RAG Assistant",
    page_icon="🌿",
    layout="wide",
)

# --- Palette Custom CSS ---
st.markdown(
    """
    <style>
    :root {
        --olive-leaf: #606c38;
        --black-forest: #283618;
        --cornsilk: #fefae0;
        --cornsilk-card: #f5efd0;
        --sunlit-clay: #dda15e;
        --copperwood: #bc6c25;
    }

    /* 1. Main App & Top Header */
    .stApp {
        background-color: var(--cornsilk) !important;
        color: var(--black-forest) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Add bottom padding to main block so raised chat input doesn't cover messages */
    .main .block-container {
        padding-bottom: 130px !important;
    }

    /* 2. Force High-Contrast Text in Main Area (Fixes white-on-cornsilk text) */
    section.main h1, section.main h2, section.main h3, 
    section.main h4, section.main h5, section.main h6,
    section.main p, section.main li, section.main label,
    section.main div[data-testid="stMarkdownContainer"] {
        color: var(--black-forest) !important;
    }

    /* Inline code blocks in main area */
    section.main code {
        background-color: rgba(96, 108, 56, 0.15) !important;
        color: var(--black-forest) !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }

    /* 3. Sidebar Theme & Inputs */
    section[data-testid="stSidebar"] {
        background-color: var(--black-forest) !important;
        border-right: 2px solid var(--olive-leaf) !important;
    }

    /* Sidebar text elements */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
        color: var(--cornsilk) !important;
    }

    /* Sidebar Text Input (Dark text on Cornsilk background so you can see what you type) */
    section[data-testid="stSidebar"] input[type="text"] {
        background-color: var(--cornsilk) !important;
        color: var(--black-forest) !important;
        border: 1px solid var(--sunlit-clay) !important;
        border-radius: 6px !important;
        caret-color: var(--black-forest) !important;
    }

    section[data-testid="stSidebar"] input[type="text"]::placeholder {
        color: rgba(40, 54, 24, 0.55) !important;
    }

    /* Sidebar File Uploader Dropzone */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(254, 250, 224, 0.08) !important;
        border: 1px dashed var(--sunlit-clay) !important;
        color: var(--cornsilk) !important;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background-color: var(--olive-leaf) !important;
        color: var(--cornsilk) !important;
        border: none !important;
    }

    /* Sidebar Info Box */
    section[data-testid="stSidebar"] div[data-testid="stAlert"] {
        background-color: rgba(96, 108, 56, 0.35) !important;
        border: 1px solid var(--sunlit-clay) !important;
        border-radius: 8px !important;
    }

    section[data-testid="stSidebar"] code {
        background-color: rgba(254, 250, 224, 0.15) !important;
        color: var(--sunlit-clay) !important;
    }

    /* 4. Primary Accent Buttons */
    .stButton > button {
        background-color: var(--copperwood) !important;
        color: var(--cornsilk) !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease-in-out;
    }

    .stButton > button:hover {
        background-color: var(--sunlit-clay) !important;
        color: var(--black-forest) !important;
        transform: translateY(-1px);
    }

    .stButton > button:hover * {
        color: var(--black-forest) !important;
    }

    /* 5. Header Badge */
    .header-tag {
        background-color: var(--olive-leaf) !important;
        color: var(--cornsilk) !important;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
        margin-right: 0.4rem;
    }

    /* 6. Chat Messages & Expanders */
    div[data-testid="stChatMessage"] {
        background-color: var(--cornsilk-card) !important;
        border: 1px solid rgba(96, 108, 56, 0.25) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 0.85rem !important;
        box-shadow: 0 2px 5px rgba(40, 54, 24, 0.05) !important;
    }

    div[data-testid="stChatMessage"] p,
    div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] span,
    div[data-testid="stChatMessage"] div {
        color: var(--black-forest) !important;
    }

    details[data-testid="stExpander"] {
        background-color: rgba(221, 161, 94, 0.12) !important;
        border: 1px solid var(--sunlit-clay) !important;
        border-radius: 8px !important;
        margin-top: 0.5rem !important;
    }

    details[data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary * {
        color: var(--black-forest) !important;
        font-weight: 600 !important;
    }

    /* 7. Lift & Style the Bottom Chat Input Box */
    div[data-testid="stBottom"] {
        background-color: transparent !important;
        padding-bottom: 2.2rem !important; /* Lifts the chat box up from the bottom edge */
    }

    div[data-testid="stBottom"] > div {
        background-color: var(--cornsilk) !important;
        padding-top: 0.5rem !important;
    }

    div[data-testid="stChatInput"] {
        background-color: #ffffff !important;
        border: 2px solid var(--olive-leaf) !important;
        border-radius: 14px !important;
        box-shadow: 0 6px 18px rgba(40, 54, 24, 0.12) !important;
        padding: 2px 6px !important;
    }

    div[data-testid="stChatInput"]:focus-within {
        border-color: var(--copperwood) !important;
        box-shadow: 0 6px 20px rgba(188, 108, 37, 0.22) !important;
    }

    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        color: var(--black-forest) !important;
        caret-color: var(--copperwood) !important;
        font-size: 1rem !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: rgba(40, 54, 24, 0.55) !important;
    }

    /* Chat Submit Button */
    div[data-testid="stChatInput"] button {
        background-color: var(--copperwood) !important;
        color: var(--cornsilk) !important;
        border-radius: 10px !important;
    }

    div[data-testid="stChatInput"] button svg {
        fill: var(--cornsilk) !important;
    }

    div[data-testid="stChatInput"] button:hover {
        background-color: var(--olive-leaf) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Initialize Session State for Performance ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "rag" not in st.session_state:
    st.session_state["rag"] = None

if "current_k" not in st.session_state:
    st.session_state["current_k"] = RETRIEVER_K

if "active_dataset_id" not in st.session_state:
    st.session_state["active_dataset_id"] = None


# --- Sidebar Controls ---
st.sidebar.markdown("## 🌿 Personal RAG")
st.sidebar.markdown(
    "<span class='header-tag'>Knowledge Base & Settings</span>",
    unsafe_allow_html=True,
)

data_option = st.sidebar.radio(
    "Data Source Mode",
    options=["Use Existing Data", "Upload CV & Add GitHub Username"],
    index=0,
)

# Parameters to determine based on mode
target_github_username = GITHUB_USERNAME
target_pdf_path = PDF_PATH
target_index_path = FAISS_INDEX_PATH
can_build = True

if data_option == "Use Existing Data":
    st.sidebar.info(
        f"**GitHub:** `{GITHUB_USERNAME}`\n\n"
        f"**CV:** `{PDF_PATH.name}`\n\n"
    )
    dataset_id = "default"
else:
    custom_username = st.sidebar.text_input(
        "GitHub Username",
        value="",
        placeholder="e.g. octocat",
    ).strip()

    uploaded_cv = st.sidebar.file_uploader(
        "Upload CV (PDF)",
        type=["pdf"],
        help="Upload your personal resume/CV in PDF format",
    )

    if custom_username:
        target_github_username = custom_username
        target_index_path = DATA_DIR / f"faiss_index_{custom_username}"
    else:
        can_build = False

    if uploaded_cv is not None:
        uploads_dir = DATA_DIR / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        custom_pdf_path = uploads_dir / uploaded_cv.name
        custom_pdf_path.write_bytes(uploaded_cv.getbuffer())
        target_pdf_path = custom_pdf_path
    else:
        can_build = False

    dataset_id = f"custom_{target_github_username}_{uploaded_cv.name if uploaded_cv else 'none'}"

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Retrieval & Sync")

# 1. K Selector
k_val = st.sidebar.slider(
    "Retriever K (Documents)",
    min_value=1,
    max_value=15,
    value=st.session_state["current_k"],
    help="Number of documents to retrieve via similarity search (CV chunks are always included).",
)

# Dynamically update K if RAG is already built
if st.session_state["rag"] is not None and k_val != st.session_state["current_k"]:
    st.session_state["rag"].update_k(k_val)
    st.session_state["current_k"] = k_val

# 2. Refresh / Force Fetch Checkbox
force_fetch_github = st.sidebar.checkbox(
    "Refresh GitHub Data (force fetch)",
    value=False,
    help="Re-download all repositories, READMEs, and code files from GitHub even if previously cached.",
)

# 3. Action Buttons
col_btn1, col_btn2 = st.sidebar.columns(2)
with col_btn1:
    rebuild_btn = st.button("🔄 Sync & Build", use_container_width=True)

with col_btn2:
    clear_chat = st.button("🗑️ Clear Chat", use_container_width=True)

if clear_chat:
    st.session_state["messages"] = []
    st.rerun()


# --- Build / Sync RAG Handler ---
def initialize_rag(force_fetch: bool = False, force_reindex: bool = False):
    """Fetch GitHub data if needed and build/load the Personal RAG instance."""
    with st.spinner(f"Ingesting GitHub data for @{target_github_username}..."):
        ingest_github(
            username=target_github_username,
            output_dir=str(GITHUB_DATA_DIR),
            force_fetch=force_fetch,
        )

    with st.spinner("Indexing documents & building conversational chain..."):
        rag = PersonalRAG(
            pdf_path=target_pdf_path,
            github_username=target_github_username,
            github_data_dir=GITHUB_DATA_DIR,
            index_path=target_index_path,
            k=k_val,
        ).build(force_reindex=force_reindex)

        st.session_state["rag"] = rag
        st.session_state["active_dataset_id"] = dataset_id
        st.session_state["current_k"] = k_val


# Trigger rebuild when button clicked
if rebuild_btn:
    if not can_build:
        st.sidebar.error("Please provide both a GitHub username and a CV PDF file.")
    else:
        initialize_rag(force_fetch=force_fetch_github, force_reindex=True)
        st.sidebar.success("Knowledge base rebuilt successfully!")
        st.rerun()

# Auto-load existing index on initial startup if not loaded yet
if st.session_state["rag"] is None and data_option == "Use Existing Data":
    if target_index_path.exists():
        with st.spinner("Loading existing knowledge base..."):
            rag = PersonalRAG(
                pdf_path=target_pdf_path,
                github_username=target_github_username,
                github_data_dir=GITHUB_DATA_DIR,
                index_path=target_index_path,
                k=k_val,
            ).build(force_reindex=False)
            st.session_state["rag"] = rag
            st.session_state["active_dataset_id"] = dataset_id
            st.session_state["current_k"] = k_val


# --- Main UI Area ---
st.title("🌿 Personal RAG Assistant")
st.markdown(
    "Query personal background, resume information, technical skills, and GitHub repositories with grounded citations."
)

# Status notification
if st.session_state["rag"] is None:
    st.warning(
        "👋 The RAG system is not initialized yet. "
        "Select your data settings in the sidebar and click **🔄 Sync & Build** to get started."
    )
else:
    active_user = st.session_state["rag"].github_username
    st.markdown(
        f"<span class='header-tag'>Active Profile: @{active_user}</span> "
        f"<span class='header-tag'>Retriever K: {st.session_state['current_k']}</span> ",
        unsafe_allow_html=True,
    )

# --- Render Chat History ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Sources & References", expanded=False):
                for doc in msg["sources"]:
                    src = doc.metadata.get("source", "N/A")
                    doc_type = doc.metadata.get("type", "document")
                    st.markdown(f"- **[{doc_type}]** `{src}`")

# --- Chat Input & Response ---
if prompt := st.chat_input("Ask a question about experience, projects, skills, or code..."):
    if st.session_state["rag"] is None:
        st.error("Please build or initialize the knowledge base in the sidebar first!")
    else:
        # Display user message
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching CV & repositories..."):
                response = st.session_state["rag"].ask(prompt)
                answer = response.get("answer", "No answer generated.")
                sources = response.get("source_documents", [])

                st.write(answer)
                if sources:
                    with st.expander("📚 Sources & References", expanded=False):
                        for doc in sources:
                            src = doc.metadata.get("source", "N/A")
                            doc_type = doc.metadata.get("type", "document")
                            st.markdown(f"- **[{doc_type}]** `{src}`")

        # Save to chat history
        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
