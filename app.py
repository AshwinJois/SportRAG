import streamlit as st
import tempfile
import base64
from data_ingestion.ingest import build_documents
from postgres.connection import list_collections
from vector_db.store_to_pgvector import store_to_db

st.set_page_config(
    page_title="SportRAG",
    page_icon="🏏",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Load background image ──────────────────────────────────────────────────────
def get_base64_image(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""
bg_image = get_base64_image("cricketIcon.jpg")
bg_css = f"background-image: url('data:image/jpeg;base64,{bg_image}');" if bg_image else "background: #0a0a0f;"

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;900&family=DM+Sans:wght@300;400;500&family=Bebas+Neue&display=swap');

  :root {{
    --gold:    #c9a84c;
    --gold-lt: #e8c96a;
    --cream:   #f5f0e8;
    --dark:    #080810;
    --card:    rgba(10,10,20,0.82);
    --border:  rgba(201,168,76,0.25);
  }}

  /* ── Nuke ALL Streamlit white backgrounds ── */
  html, body, #root,
  .stApp,
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  [data-testid="stHeader"],
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stMainBlockContainer"],
  section.main, .main, .css-1d391kg,
  .stApp > header {{
    background: transparent !important;
    background-color: transparent !important;
  }}

  /* ── Root background on html/body ── */
  html, body {{
    {bg_css}
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
    background-color: #080810 !important;
  }}

  /* ── App shell gets the image too as fallback ── */
  .stApp {{
    {bg_css}
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
    background-color: #080810;
  }}

  /* Dark vignette overlay */
  .stApp::before {{
    content: '';
    position: fixed;
    inset: 0;
    background:
      radial-gradient(ellipse at center, rgba(8,8,16,0.55) 0%, rgba(8,8,16,0.90) 100%),
      linear-gradient(180deg, rgba(8,8,16,0.6) 0%, rgba(8,8,16,0.4) 50%, rgba(8,8,16,0.8) 100%);
    pointer-events: none;
    z-index: 0;
  }}

  /* ── Main content block ── */
  .block-container {{
    position: relative;
    z-index: 1;
    max-width: 760px !important;
    padding: 3rem 2rem 4rem !important;
    background: transparent !important;
  }}

  /* ── Typography ── */
  h1 {{
    font-family: 'Playfair Display', serif !important;
    font-size: 3.4rem !important;
    font-weight: 900 !important;
    color: var(--gold-lt) !important;
    letter-spacing: -0.5px;
    line-height: 1.1 !important;
    text-align: center;
    margin-bottom: 0.25rem !important;
    text-shadow: 0 0 60px rgba(201,168,76,0.35);
  }}

  h2, h3 {{
    font-family: 'Playfair Display', serif !important;
    color: var(--cream) !important;
  }}

  /* Force all text elements to cream — covers Streamlit's deeply nested spans */
  p, li, label, div, span,
  .stMarkdown, .stMarkdown p,
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stText"],
  .stText {{
    font-family: 'DM Sans', sans-serif !important;
    color: rgba(245,240,232,0.88) !important;
  }}

  /* ── Tagline ── */
  .tagline {{
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 5px;
    font-size: 0.85rem;
    color: var(--gold);
    text-align: center;
    margin-bottom: 2rem;
    opacity: 0.85;
  }}
  
  

  /* ── Gold divider ── */
  hr {{
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 2rem 0 !important;
  }}

  /* ── Cards / section wrappers ── */
  .rag-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);
    box-shadow: 0 8px 48px rgba(0,0,0,0.55), inset 0 1px 0 rgba(201,168,76,0.12);
  }}

  .section-label {{
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 4px;
    font-size: 0.72rem;
    color: var(--gold);
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    display: block;
  }}

  /* ── Subheader override ── */
  .stApp [data-testid="stSubheader"] {{
    font-family: 'Playfair Display', serif !important;
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    color: var(--cream) !important;
    border-left: 3px solid var(--gold);
    padding-left: 0.75rem;
  }}

  /* ── Selectbox ── */
  .stSelectbox > div > div {{
    background: rgba(20,20,35,0.9) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
  }}

  /* ── Text input ── */
  .stTextInput > div > div > input {{
    background: rgba(20,20,35,0.9) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
  }}

  .stTextInput > div > div > input:focus {{
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 2px rgba(201,168,76,0.2) !important;
  }}

  /* ── File uploader ── */
  [data-testid="stFileUploader"],
  [data-testid="stFileUploader"] > div,
  [data-testid="stFileUploaderDropzone"] {{
    background: rgba(20,20,35,0.85) !important;
    border: 1.5px dashed var(--gold) !important;
    border-radius: 10px !important;
  }}

  [data-testid="stFileUploaderDropzone"] {{
    padding: 1.5rem !important;
  }}

  [data-testid="stFileUploader"]:hover,
  [data-testid="stFileUploaderDropzone"]:hover {{
    border-color: var(--gold-lt) !important;
    background: rgba(30,25,50,0.9) !important;
  }}

  /* All text INSIDE the uploader */
  [data-testid="stFileUploaderDropzone"] *,
  [data-testid="stFileUploaderDropzone"] span,
  [data-testid="stFileUploaderDropzone"] p,
  [data-testid="stFileUploaderDropzone"] small,
  [data-testid="stFileUploader"] label,
  [data-testid="stFileUploader"] span,
  [data-testid="stFileUploader"] p {{
    color: var(--cream) !important;
  }}

  /* The "Browse files" button inside uploader */
  [data-testid="stFileUploaderDropzone"] button,
  [data-testid="baseButton-secondary"] {{
    background: rgba(201,168,76,0.15) !important;
    border: 1px solid var(--gold) !important;
    color: var(--gold-lt) !important;
    border-radius: 6px !important;
  }}

  /* Widget labels above inputs */
  [data-testid="stWidgetLabel"],
  [data-testid="stWidgetLabel"] p,
  .stSelectbox label,
  .stTextInput label,
  .stRadio label,
  .stFileUploader label {{
    color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
  }}

  /* ── Primary button ── */
  .stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, var(--gold) 0%, #a8832a 100%) !important;
    color: #0a0a0f !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1rem !important;
    letter-spacing: 3px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 24px rgba(201,168,76,0.3) !important;
  }}

  .stButton > button[kind="primary"]:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(201,168,76,0.5) !important;
    background: linear-gradient(135deg, var(--gold-lt) 0%, var(--gold) 100%) !important;
  }}

  /* ── Secondary / default button ── */
  .stButton > button:not([kind="primary"]) {{
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
  }}

  .stButton > button:not([kind="primary"]):hover {{
    border-color: var(--gold) !important;
    color: var(--gold-lt) !important;
  }}

  /* ── Radio buttons ── */
  .stRadio > div {{
    gap: 0.5rem !important;
  }}

  .stRadio label {{
    background: rgba(20,20,35,0.7) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 0.5rem 1rem !important;
    transition: border-color 0.2s ease !important;
  }}

  .stRadio label:hover {{
    border-color: var(--gold) !important;
  }}

  /* ── Info / warning / status boxes ── */
  .stInfo, [data-testid="stInfoBox"] {{
    background: rgba(201,168,76,0.08) !important;
    border: 1px solid rgba(201,168,76,0.3) !important;
    border-radius: 8px !important;
    color: var(--cream) !important;
  }}

  .stWarning, [data-testid="stWarningBox"] {{
    background: rgba(201,168,76,0.1) !important;
    border: 1px solid rgba(201,168,76,0.4) !important;
    border-radius: 8px !important;
  }}

  /* ── Status / spinner ── */
  [data-testid="stStatusWidget"] {{
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
  }}

  /* ── Spinner text ── */
  .stSpinner > div {{
    border-top-color: var(--gold) !important;
  }}
  
  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: var(--gold); }}
    
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "ingestion_done" not in st.session_state:
    st.session_state.ingestion_done = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieved_images" not in st.session_state:
    st.session_state.retrieved_images = []
if "help_visible" not in st.session_state:
    st.session_state.help_visible = False
if st.session_state.ingestion_done:
    st.switch_page("pages/chat.py")

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.title("🏏 SportRAG")
st.markdown('<p class="tagline">Intelligence. Precision. Performance.</p>', unsafe_allow_html=True)

st.markdown(
    """
    <p style="max-width: 720px;">
        SportRag is a Multimodal RAG with Cricket related knowledgebase. Users can select an existing collection
        or can upload a new document. The SportRAG responds to user's Cricket related questions with Images if any.
        The RAG pipeline can also be evaluated using RAGAS framework.
    </p>
    """,
    unsafe_allow_html=True
)
# ── Help guide button ─────────────────────────────────────────────────────
HELP_GUIDE_TEXT = """
SportRAG is a multimodal Retrieval-Augmented Generation app focused on cricket intelligence. It combines curated
knowledge collections with new document ingestion, letting you surface precise insights plus reference images.

How to use:
1. Open an existing collection from the dropdown to query past uploads.
2. Or upload a new PDF, choose whether to append or create a collection, and hit "Process Document".
3. Once ingestion finishes you are redirected to the chat, where you can ask cricket questions and get text/image answers.
4. Use the RAGAS evaluation workflow in the chat page if you need to benchmark the pipeline.
"""
if st.button("Help Guide", type="secondary", use_container_width=True):
    st.session_state.help_visible = not st.session_state.help_visible
if st.session_state.help_visible:
    st.info(HELP_GUIDE_TEXT, icon="ℹ️")
st.divider()

# ── Section 1: Existing Collections ───────────────────────────────────────────
st.subheader("📂 Open Existing Collection")
collections = list_collections()
if collections:
    selected = st.selectbox(
        "Choose a collection to query:",
        options=collections,
        index=None,
        placeholder="Select a collection...",
        label_visibility="collapsed"
    )
    if st.button("Open Collection", type="primary", use_container_width=True, disabled=not selected):
        st.session_state.collection_name = selected
        st.session_state.uploaded_filename = selected
        st.session_state.ingestion_done = True
        st.session_state.messages = []
        st.session_state.retrieved_images = []
        st.switch_page("pages/chat.py")
else:
    st.info("No existing collections found in the database.")
st.markdown("</div>", unsafe_allow_html=True)
st.divider()

# ── Section 2: Upload New Document ────────────────────────────────────────────
st.subheader("📤 Upload New Document")
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    label_visibility="collapsed"
)
if uploaded_file:
    st.info(f"📄 **{uploaded_file.name}** ready to process.")
    st.markdown("**Where would you like to store this document?**")
    storage_choice = st.radio(
        "Collection option",
        options=["Add to existing collection", "Create new collection"],
        label_visibility="collapsed"
    )
    if storage_choice == "Add to existing collection":
        if collections:
            collection_name_input = st.selectbox(
                "Select collection to add to:",
                options=collections
            )
        else:
            st.warning("No existing collections found. Please create a new one.")
            collection_name_input = None
    else:
        default_name = uploaded_file.name.replace(".pdf", "").lower().replace(" ", "_")
        collection_name_input = st.text_input(
            "New collection name",
            value=default_name,
            help="Only lowercase letters, numbers and underscores."
        )
        if collection_name_input in collections:
            st.warning(f"⚠️ **{collection_name_input}** already exists. Choose a different name or add to the existing one.")
            collection_name_input = None
    if collection_name_input and st.button("Process Document", type="primary", use_container_width=True):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        with st.status("Processing document...", expanded=True) as status:
            st.write("📑 Extracting text and images...")
            all_documents = build_documents(tmp_path)
            st.write(f"✅ Extracted {len(all_documents)} chunks. Storing to **{collection_name_input}**...")
            store_to_db(all_documents, collection_name_input)
            status.update(label="Done! Redirecting to chat...", state="complete")
            st.session_state.ingested_documents = all_documents
        st.session_state.collection_name = collection_name_input
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.ingestion_done = True
        st.session_state.messages = []
        st.session_state.retrieved_images = []
        st.switch_page("pages/chat.py")
st.markdown("</div>", unsafe_allow_html=True)
