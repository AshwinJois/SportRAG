import base64
import streamlit as st
from evaluation.RAGAS import run_rag_evaluation, run_ragas_evaluation
from evaluation.synthetic_data_generation import generate_synthetic_dataset
from postgres.connection import create_evaluation_table, ensure_evaluation_table, insert_ragas_result

st.set_page_config(
    page_title="SportRAG – Evaluation",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_base64_image(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

bg_image = get_base64_image("cricketIcon.jpg")
bg_css = f"background-image: url('data:image/jpeg;base64,{bg_image}');" if bg_image else "background: #080810;"
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;0,700;0,900;1,400&family=DM+Sans:wght@300;400;500&family=Bebas+Neue&display=swap');

  :root {{
    --gold:    #c9a84c;
    --gold-lt: #e8c96a;
    --cream:   #f5f0e8;
    --dark:    #080810;
    --card:    rgba(10,10,20,0.82);
    --border:  rgba(201,168,76,0.25);
    --border-bright: rgba(201,168,76,0.5);
    --user-bg: rgba(201,168,76,0.10);
    --ai-bg:   rgba(20,20,40,0.75);
    --step-inactive: rgba(201,168,76,0.06);
    --step-active: rgba(201,168,76,0.12);
  }}

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

  html, body {{
    {bg_css}
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
    background-color: #080810 !important;
  }}

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
    background: radial-gradient(ellipse at 30% 50%, rgba(8,8,16,0.5) 0%, rgba(8,8,16,0.95) 100%);
    pointer-events: none;
    z-index: 0;
  }}

  .block-container {{
    position: relative;
    z-index: 1;
    padding: 2rem 3rem 6rem !important;
    background: transparent !important;
    max-width: 1100px !important;
  }}

  * {{ font-family: 'DM Sans', sans-serif; }}

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

  /* ── Hero Header ── */
  .hero-wrap {{
    text-align: center;
    padding: 2.5rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2.5rem;
  }}
  .hero-eyebrow {{
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 6px;
    font-size: 0.7rem;
    color: var(--gold);
    opacity: 0.75;
    margin-bottom: 0.5rem;
  }}
  .hero-title {{
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    color: var(--gold-lt);
    text-shadow: 0 0 60px rgba(201,168,76,0.25);
    margin: 0 0 0.6rem;
    line-height: 1.1;
  }}
  .hero-subtitle {{
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: rgba(245,240,232,0.55);
    max-width: 560px;
    margin: 0 auto;
    line-height: 1.6;
  }}

  /* ── Pipeline connector ── */
  .pipeline-connector {{
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0.6rem 0;
    gap: 0.5rem;
  }}
  .pipeline-line {{
    height: 1px;
    width: 60px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    opacity: 0.4;
  }}
  .pipeline-arrow {{
    color: var(--gold);
    font-size: 1.1rem;
    opacity: 0.6;
  }}

  /* ── Step Cards ── */
  .step-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    backdrop-filter: blur(16px);
    position: relative;
    overflow: hidden;
    margin-bottom: 0.25rem;
    transition: border-color 0.3s ease;
  }}
  .step-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    opacity: 0.5;
  }}
  .step-card:hover {{
    border-color: var(--border-bright);
  }}
  .step-header {{
    display: flex;
    align-items: flex-start;
    gap: 1.2rem;
    margin-bottom: 1rem;
  }}
  .step-number {{
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.5rem;
    line-height: 1;
    color: var(--gold);
    opacity: 0.25;
    flex-shrink: 0;
    margin-top: -4px;
  }}
  .step-title {{
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--gold-lt);
    margin: 0 0 0.3rem;
  }}
  .step-desc {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.88rem;
    color: rgba(245,240,232,0.6);
    line-height: 1.65;
    margin: 0;
  }}
  .step-badge-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1.2rem;
  }}
  .step-badge {{
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 2px;
    font-size: 0.62rem;
    background: rgba(201,168,76,0.1);
    border: 1px solid rgba(201,168,76,0.25);
    color: var(--gold);
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
  }}
  .step-divider {{
    height: 1px;
    background: var(--border);
    margin: 1.2rem 0;
  }}

  /* ── Results ── */
  .results-wrap {{
    background: rgba(8,8,16,0.6);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.2rem;
  }}
  .metric-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-top: 1rem;
  }}
  .metric-tile {{
    background: rgba(201,168,76,0.06);
    border: 1px solid rgba(201,168,76,0.2);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
  }}
  .metric-name {{
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 2px;
    font-size: 0.65rem;
    color: var(--gold);
    opacity: 0.8;
    margin-bottom: 0.4rem;
  }}
  .metric-score {{
    font-family: 'Playfair Display', serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--gold-lt);
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
    background: rgba(8,8,16,0.92) !important;
    border-right: 1px solid var(--border) !important;
    backdrop-filter: blur(20px) !important;
  }}
  [data-testid="stSidebar"] * {{ color: var(--cream) !important; }}
  [data-testid="stSidebar"] hr {{ border-color: var(--border) !important; }}

  /* ── Buttons ── */
  .stButton > button {{
    background: linear-gradient(135deg, var(--gold) 0%, #a8832a 100%) !important;
    color: #0a0a0f !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 0.9rem !important;
    letter-spacing: 3px !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 1.8rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(201,168,76,0.25) !important;
  }}
  .stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(201,168,76,0.45) !important;
    background: linear-gradient(135deg, var(--gold-lt) 0%, var(--gold) 100%) !important;
  }}

  /* ── Misc ── */
  .stSpinner > div {{ border-top-color: var(--gold) !important; }}
  hr {{ border: none !important; border-top: 1px solid var(--border) !important; }}
  ::-webkit-scrollbar {{ width: 5px; }}
  ::-webkit-scrollbar-track {{ background: transparent; }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: var(--gold); }}
  .stWarning {{ background: rgba(201,168,76,0.1) !important; border-color: var(--gold) !important; }}
  .stInfo    {{ background: rgba(201,168,76,0.06) !important; border-color: var(--border) !important; }}
  .stSuccess {{ background: rgba(40,80,40,0.3) !important; border-color: rgba(100,200,100,0.3) !important; }}
  [data-testid="stExpander"] {{
    background: rgba(10,10,20,0.6) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
  }}
  .stDataFrame {{ background: transparent !important; }}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-eyebrow">RAG · Quality Assurance · Pipeline</div>
  <h1 class="hero-title">🏏 SportRAG Evaluation</h1>
  <p class="hero-subtitle">
    A three-stage pipeline to measure how well your RAG system retrieves,
    reasons, and responds — powered by synthetic data and RAGAS metrics.
  </p>
</div>
""", unsafe_allow_html=True)

# ── STEP 1: Synthetic Data Generation ────────────────────────────────────────
st.markdown("""
<div class="step-card">
  <div class="step-header">
    <div class="step-number">01</div>
    <div>
      <div class="step-title">Synthetic Dataset Generation</div>
      <p class="step-desc">
        Samples document chunks from your vector store and prompts the LLM to synthesise
        realistic question–answer pairs grounded in the source text. Each pair is stored
        alongside its context chunk so downstream evaluation remains traceable to the
        original documents.
      </p>
    </div>
  </div>
  <div class="step-badge-row">
    <span class="step-badge">100 QA Pairs</span>
    <span class="step-badge">LLM-Generated</span>
    <span class="step-badge">Postgres · evaluation_dataset</span>
    <span class="step-badge">Columns: question · ground_truth · context</span>
  </div>
  <div class="step-divider"></div>
""", unsafe_allow_html=True)

if st.button("Generate Synthetic Dataset", key="btn_synth"):
    ensure_evaluation_table()
    st.info("✅ Evaluation table is ready.")
    with st.spinner("Asking the LLM to generate QA pairs from your document chunks…"):
        total = generate_synthetic_dataset(target_count=100)
    st.success(f"✅ {total} synthetic QA pairs written to `evaluation_dataset`.")
st.markdown("</div>", unsafe_allow_html=True)

# ── Connector ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pipeline-connector">
  <div class="pipeline-line"></div>
  <div class="pipeline-arrow">▼</div>
  <div class="pipeline-line"></div>
</div>
""", unsafe_allow_html=True)

# ── STEP 2: RAG Pipeline Evaluation ──────────────────────────────────────────
st.markdown("""
<div class="step-card">
  <div class="step-header">
    <div class="step-number">02</div>
    <div>
      <div class="step-title">RAG Pipeline Run</div>
      <p class="step-desc">
        Takes the 100 synthetic questions and feeds each one through your full RAG
        pipeline — retrieval, reranking, and generation. The model's answer is stored in
        the <em>rag_response</em> column, completing the dataset needed for metric scoring
        in the next step.
      </p>
    </div>
  </div>
  <div class="step-badge-row">
    <span class="step-badge">100 Questions</span>
    <span class="step-badge">Full Retrieval + Generation</span>
    <span class="step-badge">Postgres · rag_response column</span>
    <span class="step-badge">Prerequisite: Step 01</span>
  </div>
  <div class="step-divider"></div>
""", unsafe_allow_html=True)

if st.button("Run RAG Pipeline", key="btn_rag"):
    with st.spinner("Running your RAG pipeline across all 100 questions…"):
        run_rag_evaluation()
    st.success("✅ RAG responses saved to `evaluation_dataset.rag_response`.")
st.markdown("</div>", unsafe_allow_html=True)

# ── Connector ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="pipeline-connector">
  <div class="pipeline-line"></div>
  <div class="pipeline-arrow">▼</div>
  <div class="pipeline-line"></div>
</div>
""", unsafe_allow_html=True)

# ── STEP 3: RAGAS Evaluation ──────────────────────────────────────────────────
st.markdown("""
<div class="step-card">
  <div class="step-header">
    <div class="step-number">03</div>
    <div>
      <div class="step-title">RAGAS Metric Scoring</div>
      <p class="step-desc">
        Runs four RAGAS metrics across all QA pairs to produce a rigorous, multi-dimensional
        picture of your RAG system's quality. Each metric targets a distinct failure mode —
        hallucination, irrelevance, noisy retrieval, and incomplete context coverage.
      </p>
    </div>
  </div>
  <div class="step-badge-row">
    <span class="step-badge">Faithfulness</span>
    <span class="step-badge">Answer Relevancy</span>
    <span class="step-badge">Context Precision</span>
    <span class="step-badge">Context Recall</span>
    <span class="step-badge">Prerequisite: Steps 01 + 02</span>
  </div>
  <div class="step-divider"></div>
""", unsafe_allow_html=True)

# Metric descriptions
col1, col2, col3, col4 = st.columns(4)
metrics_info = [
    ("Faithfulness", "Is the answer fully grounded in the retrieved context? Detects hallucinations."),
    ("Answer Relevancy", "Does the answer actually address what was asked? Flags vague or off-topic responses."),
    ("Context Precision", "Is the retrieved context free of noise? High precision means no irrelevant chunks."),
    ("Context Recall", "Does the context contain everything needed to answer? Low recall means missing chunks."),
]
for col, (name, desc) in zip([col1, col2, col3, col4], metrics_info):
    with col:
        st.markdown(f"""
        <div style="background:rgba(201,168,76,0.06);border:1px solid rgba(201,168,76,0.18);
                    border-radius:10px;padding:0.9rem 1rem;height:100%;margin-bottom:1rem;">
          <div style="font-family:'Bebas Neue',sans-serif;letter-spacing:2px;font-size:0.65rem;
                      color:var(--gold);margin-bottom:0.4rem;">{name}</div>
          <div style="font-size:0.8rem;color:rgba(245,240,232,0.55);line-height:1.55;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)
st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
if st.button("Run RAGAS Evaluation", key="btn_ragas"):
    with st.spinner("Scoring all 400 metric × question combinations — this may take a few minutes…"):
        try:
            result = run_ragas_evaluation()
            st.success("✅ RAGAS evaluation complete!")
            result_df = result.to_pandas()
            metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
            # Filter to only metric cols that exist (safety)
            available = [c for c in metric_cols if c in result_df.columns]
            avg_scores = result_df[available].mean()
            # ✅ Pass actual float values, not column name strings
            insert_ragas_result(
                faithfulness=float(avg_scores["faithfulness"]),
                answer_relevancy=float(avg_scores["answer_relevancy"]),
                context_precision=float(avg_scores["context_precision"]),
                context_recall=float(avg_scores["context_recall"]),
            )
            st.success("✅ Stored RAGAS results!")
            # Render custom metric tiles
            tile_html = '<div class="metric-grid">'
            labels = {
                "faithfulness": "Faithfulness",
                "answer_relevancy": "Answer Relevancy",
                "context_precision": "Context Precision",
                "context_recall": "Context Recall",
            }
            for col in available:
                score = avg_scores[col]
                color = "#6fcf97" if score >= 0.7 else ("#f2c94c" if score >= 0.5 else "#eb5757")
                tile_html += f"""
                <div class="metric-tile">
                  <div class="metric-name">{labels.get(col, col)}</div>
                  <div class="metric-score" style="color:{color};">{score:.3f}</div>
                </div>"""
            tile_html += "</div>"
            st.markdown(f"""
            <div class="results-wrap">
              <div style="font-family:'Bebas Neue',sans-serif;letter-spacing:3px;font-size:0.7rem;
                          color:var(--gold);opacity:0.8;margin-bottom:0.2rem;">Average Scores · 100 Questions</div>
              <div style="font-family:'Playfair Display',serif;font-size:1.2rem;
                          color:var(--gold-lt);margin-bottom:0.5rem;">RAGAS Results</div>
              {tile_html}
            </div>
            """, unsafe_allow_html=True)
            with st.expander("📋 Full per-question breakdown"):
                st.dataframe(result_df, use_container_width=True)
        except ValueError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
if st.button("Show RAGAS Results", key="btn_show_ragas"):
    try:
        from postgres.connection import fetch_ragas_results
        df = fetch_ragas_results()
        if df.empty:
            st.warning("No results found. Run RAGAS Evaluation first.")
        else:
            # Take the most recent row for the metric tiles
            latest = df.iloc[0]
            tile_html = '<div class="metric-grid">'
            labels = {
                "faithfulness": "Faithfulness",
                "answer_relevancy": "Answer Relevancy",
                "context_precision": "Context Precision",
                "context_recall": "Context Recall",
            }
            for col, label in labels.items():
                score = latest[col]
                color = "#6fcf97" if score >= 0.7 else ("#f2c94c" if score >= 0.5 else "#eb5757")
                tile_html += f"""
                <div class="metric-tile">
                  <div class="metric-name">{label}</div>
                  <div class="metric-score" style="color:{color};">{score:.3f}</div>
                </div>"""
            tile_html += "</div>"
            st.markdown(f"""
            <div class="results-wrap">
              <div style="font-family:'Bebas Neue',sans-serif;letter-spacing:3px;font-size:0.7rem;
                          color:var(--gold);opacity:0.8;">Latest Run </div>
              <div style="font-family:'Playfair Display',serif;font-size:1.2rem;
                          color:var(--gold-lt);margin-bottom:0.5rem;">RAGAS Results</div>
              {tile_html}
            </div>
            """, unsafe_allow_html=True)
            # Show history if there are multiple runs
            if len(df) > 1:
                with st.expander(f"📈 History — {len(df)} evaluation runs"):
                    st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Could not load results: {e}")
st.markdown("</div>", unsafe_allow_html=True)