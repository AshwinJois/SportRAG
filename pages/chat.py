import base64
import re
import uuid
import streamlit as st
from llm_model.model import response_generation_stream, get_retrieved_image_paths, openai_rerank_pointwise, build_agent
from postgres.plot_embeddings import get_query_plot_data
from vector_db.store_to_pgvector import get_vector_store
import plotly.express as px

st.set_page_config(
    page_title="SportRAG – Chat",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)
with st.sidebar:
    st.markdown(
        """
        <h1 style="
            font-family: 'Playfair Display', serif;
            font-size: 1.8rem;
            font-weight: 900;
            color: #e8c96a;
            margin-bottom:0;
        ">
        🏏 SportRAG
        </h1>
        <p style="
            font-family: 'Bebas Neue', sans-serif;
            letter-spacing:3px;
            font-size:0.7rem;
            color:#c9a84c;
            margin-top:-5px;
        ">
        Intelligence. Precision. Performance.
        </p>
        <hr style="border-color: rgba(201,168,76,0.25)">
        """,
        unsafe_allow_html=True
    )

# ── Background ─────────────────────────────────────────────────────────────────
def get_base64_image(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

bg_image = get_base64_image("cricketIcon.jpg")
bg_css = f"background-image: url('data:image/jpeg;base64,{bg_image}');" if bg_image else "background: #080810;"
_GREETING_PHRASES = {
    "hi",
    "hello",
    "hey",
    "hiya",
    "greetings",
    "good morning",
    "good afternoon",
    "good evening",
    "who are you?"
    "howdy",
    "hey there",
    "hi there",
    "hello there",
}

def is_generic_greeting(message: str) -> bool:
    if not message:
        return False
    cleaned = re.sub(r"[^a-zA-Z\s]", "", message).strip().lower()
    normalized = " ".join(cleaned.split())
    return normalized in _GREETING_PHRASES

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
    --user-bg: rgba(201,168,76,0.10);
    --ai-bg:   rgba(20,20,40,0.75);
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
  section.main, .main {{
    background: transparent !important;
    background-color: transparent !important;
  }}

    

  /* ── Root background ── */
  html, body {{
    {bg_css}
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
    background-color: #080810 !important;
  }}

  /* ── App shell ── */
  .stApp {{
    {bg_css}
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-repeat: no-repeat;
    background-color: #080810;
  }}

  .stApp::before {{
    content: '';
    position: fixed;
    inset: 0;
    background:
      radial-gradient(ellipse at 30% 50%, rgba(8,8,16,0.5) 0%, rgba(8,8,16,0.92) 100%);
    pointer-events: none;
    z-index: 0;
  }}

  .block-container {{
    position: relative;
    z-index: 1;
    padding: 1.5rem 2rem 6rem !important;
    background: transparent !important;
  }}

  /* ── Global fonts ── */
  * {{
    font-family: 'DM Sans', sans-serif;
  }}

  h1, h2, h3 {{
    font-family: 'Playfair Display', serif !important;
    color: var(--gold-lt) !important;
  }}

  p, li, label, div, span,
  .stMarkdown, .stMarkdown p,
  [data-testid="stMarkdownContainer"] p,
  [data-testid="stText"] {{
    color: rgba(245,240,232,0.88) !important;
  }}

  /* ── Page title ── */
  .chat-title {{
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 900;
    color: var(--gold-lt);
    text-shadow: 0 0 40px rgba(201,168,76,0.3);
    margin-bottom: 0.1rem;
  }}

  .chat-subtitle {{
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 4px;
    font-size: 0.7rem;
    color: var(--gold);
    opacity: 0.75;
    margin-bottom: 1.5rem;
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
    background: rgba(8,8,16,0.88) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: blur(20px) !important;
  }}

  [data-testid="stSidebar"] * {{
    color: var(--cream) !important;
  }}

  [data-testid="stSidebar"] hr {{
    border-color: var(--border) !important;
  }}

  .sidebar-doc-label {{
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 3px;
    font-size: 0.65rem;
    color: var(--gold);
    text-transform: uppercase;
    display: block;
    margin-bottom: 0.3rem;
    opacity: 0.8;
  }}

  .sidebar-doc-name {{
    font-family: 'Playfair Display', serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--cream);
  }}

  /* ── Chat messages ── */
  [data-testid="stChatMessage"] {{
    background: var(--ai-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 0.8rem !important;
    backdrop-filter: blur(12px) !important;
    padding: 1rem 1.2rem !important;
  }}

  /* User message */
  [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{
    background: var(--user-bg) !important;
    border-color: rgba(201,168,76,0.4) !important;
  }}

  [data-testid="stChatMessage"] p {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.65 !important;
    color: var(--cream) !important;
  }}

  /* ── Chat input container ── */
  [data-testid="stChatInput"],
  [data-testid="stChatInputContainer"],
  .stChatInput {{
    background: rgba(10,10,20,0.92) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(20px) !important;
  }}

  [data-testid="stChatInput"]:focus-within,
  [data-testid="stChatInputContainer"]:focus-within {{
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(201,168,76,0.15) !important;
  }}

  /* Textarea — needs explicit dark bg + light text */
  [data-testid="stChatInput"] textarea,
  [data-testid="stChatInputContainer"] textarea,
  .stChatInput textarea,
  div[data-baseweb="textarea"] textarea,
  div[data-baseweb="base-input"] textarea {{
    color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    background: transparent !important;
    caret-color: var(--gold) !important;
  }}

  /* Placeholder text */
  [data-testid="stChatInput"] textarea::placeholder,
  [data-testid="stChatInputContainer"] textarea::placeholder {{
    color: rgba(201,168,76,0.45) !important;
  }}

  /* The baseweb wrapper inside chat input */
  div[data-baseweb="textarea"],
  div[data-baseweb="base-input"] {{
    background: transparent !important;
    border: none !important;
  }}

  /* ── Send button inside chat input ── */
  [data-testid="stChatInput"] button,
  [data-testid="stChatInputContainer"] button {{
    color: var(--gold) !important;
    background: transparent !important;
  }}

  /* ── Primary button ── */
  .stButton > button[kind="primary"],
  .stButton > button {{
    background: linear-gradient(135deg, var(--gold) 0%, #a8832a 100%) !important;
    color: #0a0a0f !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 0.9rem !important;
    letter-spacing: 3px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(201,168,76,0.25) !important;
  }}

  .stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(201,168,76,0.45) !important;
    background: linear-gradient(135deg, var(--gold-lt) 0%, var(--gold) 100%) !important;
  }}

  /* ── Spinner ── */
  .stSpinner > div {{
    border-top-color: var(--gold) !important;
  }}

  /* ── Plotly chart background ── */
  .js-plotly-plot .plotly .bg {{
    fill: transparent !important;
  }}

  /* ── Images ── */
  img {{
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5) !important;
  }}

  /* ── Divider ── */
  hr {{
    border: none !important;
    border-top: 1px solid var(--border) !important;
  }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 5px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: var(--gold); }}

  /* ── Warning / Info ── */
  .stWarning {{ background: rgba(201,168,76,0.1) !important; border-color: var(--gold) !important; }}
  .stInfo    {{ background: rgba(201,168,76,0.06) !important; border-color: var(--border) !important; }}
</style>
""", unsafe_allow_html=True)

# ── Guard ──────────────────────────────────────────────────────────────────────
if not st.session_state.get("ingestion_done"):
    st.warning("No document ingested yet. Please upload a PDF first.")
    if st.button("Go to Upload"):
        st.switch_page("app.py")
    st.stop()

# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieved_images" not in st.session_state:
    st.session_state.retrieved_images = []
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span class="sidebar-doc-label">Embedding Space</span>', unsafe_allow_html=True)
    sidebar_plot_placeholder = st.empty()
    st.caption("Points close in the original embedding space may appear far apart in 2D")
    st.divider()
    if st.button("⬅ Upload New Document", use_container_width=True):
        st.session_state.ingestion_done = False
        st.session_state.messages = []
        st.session_state.retrieved_images = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.switch_page("app.py")
    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.retrieved_images = []
        st.session_state.thread_id = str(uuid.uuid4())
    if st.button("Evaluation", use_container_width=True):
        st.switch_page("pages/evaluation.py")
    st.divider()
    st.markdown("""
            <span class="sidebar-doc-label">Active Collection</span>
        """, unsafe_allow_html=True)
    st.markdown(f"""
            <span class="sidebar-doc-name">📄 {st.session_state.get('uploaded_filename', 'Document')}</span>
        """, unsafe_allow_html=True)

# ── Main area header ───────────────────────────────────────────────────────────
st.markdown('<div class="chat-title">🏏 SportRAG</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-subtitle">Your intelligent cricket knowledge assistant</div>', unsafe_allow_html=True)
st.divider()

# ── Chat history ───────────────────────────────────────────────────────────────
img_idx = 0
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if img_idx < len(st.session_state.retrieved_images):
                paths = st.session_state.retrieved_images[img_idx]
                if paths:
                    st.image(paths, width=450)
            img_idx += 1

# ── New input ──────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask me anything about the document..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    image_paths = []
    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            vector_store = get_vector_store(st.session_state.collection_name)
            docs = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 5,
                    "score_threshold": 0.3
                }).invoke(user_input)
            rerank = openai_rerank_pointwise(user_input, docs)
            # Embedding visualisation in sidebar
            with sidebar_plot_placeholder.container():
                with st.spinner("Rendering..."):
                    plot_df = get_query_plot_data(
                        st.session_state.collection_name,
                        user_input,
                        docs
                    )
                    plot_df["snippet"] = plot_df["snippet"].str.slice(0, 200)
                    plot_df["text"] = ""
                    plot_df.loc[plot_df["label"] != "Other Chunks", "text"] = plot_df["label"]
                    fig = px.scatter(
                        plot_df,
                        x="x", y="y",
                        color="label",
                        hover_data=["snippet"],
                        color_discrete_map={
                            "Other Chunks":       "#3a3a5c",
                            "Retrieved Context":  "#c9a84c",
                            "Current Query":      "#ef4444"
                        }
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#c9a84c", family="DM Sans"),
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        legend=dict(
                            font=dict(color="#f5f0e8", size=10),
                            bgcolor="rgba(10,10,20,0.6)",
                            bordercolor="rgba(201,168,76,0.3)",
                            borderwidth=1
                        ),
                        legend_title_text="",
                        margin=dict(l=0, r=0, t=10, b=0)
                    )
                    fig.update_traces(marker=dict(size=8, opacity=0.85))
                    st.plotly_chart(fig, use_container_width=True)
            if not is_generic_greeting(user_input):
                image_paths = get_retrieved_image_paths(rerank)
        response = response_generation_stream(
            st.session_state.agent,
            rerank,
            user_input,
            st.session_state.thread_id
        )
        final_response = st.write_stream(response)
        if image_paths:
            st.image(image_paths, width=450)
    st.session_state.messages.append({"role": "assistant", "content": final_response})
    st.session_state.retrieved_images.append(image_paths)