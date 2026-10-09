import os
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st
import time
from agents.graph import RAGGraph
from utils.config import GROQ_API_KEY, TAVILY_API_KEY, COHERE_API_KEY, DATA_RAW_DIR
from rag.vectordb import VectorDB
from rag.ingest import IngestionPipeline
from eval.evaluator import RagasEvaluator

# ── cached resources ──────────────────────────────────────────────────────────



@st.cache_resource
def get_vector_db():
    print(">>> loading VectorDB", flush=True)
    db = VectorDB()
    print(">>> VectorDB done", flush=True)
    return db

@st.cache_resource
def get_ingestion_pipeline():
    print(">>> loading IngestionPipeline", flush=True)
    p = IngestionPipeline()
    print(">>> IngestionPipeline done", flush=True)
    return p

@st.cache_resource
def get_rag_graph():
    print(">>> loading RAGGraph", flush=True)
    g = RAGGraph()
    print(">>> RAGGraph done", flush=True)
    return g

@st.cache_resource
def get_evaluator():
    return RagasEvaluator()

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* ── reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 15px;
}

/* page background */
.stApp {
    background: #f8f7f5;
    color: #1a1a1a;
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e8e5e0;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.25rem;
}

/* ── main content ── */
[data-testid="stMainBlockContainer"] {
    padding: 2rem 2.5rem 1.5rem;
    max-width: 900px;
    margin: 0 auto;
}

/* ── typography helpers ── */
.page-title {
    font-size: 1.35rem;
    font-weight: 600;
    color: #1a1a1a;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem;
}
.page-sub {
    font-size: 0.875rem;
    color: #6b7280;
    margin-bottom: 1.75rem;
}
.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
    margin-bottom: 0.6rem;
}

/* ── status dots ── */
.status-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    color: #374151;
    padding: 4px 0;
}
.dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-on  { background: #22c55e; }
.dot-off { background: #d1d5db; }

/* ── source chips ── */
.source-chip {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    color: #374151;
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 5px 10px;
    margin-bottom: 5px;
    word-break: break-all;
}

/* ── pipeline steps ── */
.step {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.8rem;
    padding: 7px 10px;
    border-radius: 7px;
    border: 1px solid #e5e7eb;
    background: #f9fafb;
    margin-bottom: 5px;
    color: #6b7280;
}
.step-done {
    border-color: #bbf7d0;
    background: #f0fdf4;
    color: #15803d;
}
.step-active {
    border-color: #bfdbfe;
    background: #eff6ff;
    color: #1d4ed8;
}
.step-icon { font-size: 14px; }

/* ── chat messages ── */
.chat-wrap {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    padding-bottom: 0.5rem;
}
.msg-bot, .msg-user {
    display: flex;
    flex-direction: column;
    max-width: 88%;
}
.msg-user { align-self: flex-end; align-items: flex-end; }
.msg-bot  { align-self: flex-start; align-items: flex-start; }

.bubble {
    padding: 10px 14px;
    border-radius: 12px;
    line-height: 1.65;
    font-size: 0.875rem;
}
.bubble-user {
    background: #eff6ff;
    color: #1e40af;
    border-radius: 12px 12px 3px 12px;
    border: 1px solid #dbeafe;
}
.bubble-bot {
    background: #ffffff;
    color: #1a1a1a;
    border-radius: 3px 12px 12px 12px;
    border: 1px solid #e5e7eb;
}
.msg-meta {
    font-size: 0.7rem;
    color: #9ca3af;
    margin-top: 4px;
    padding: 0 3px;
}
.src-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 6px;
    padding: 0 3px;
}
.src-tag {
    font-size: 0.7rem;
    color: #374151;
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 4px;
    padding: 2px 7px;
}

/* ── metrics bar ── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 1.25rem;
}
.metric-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 10px 13px;
}
.metric-name  { font-size: 0.68rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 3px; }
.metric-val   { font-size: 1.25rem; font-weight: 600; }
.metric-hint  { font-size: 0.68rem; color: #9ca3af; margin-top: 1px; }
.val-good     { color: #15803d; }
.val-mid      { color: #b45309; }
.val-bad      { color: #b91c1c; }

/* ── text input ── */
.stTextInput > div > div > input {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    color: #1a1a1a !important;
    font-size: 0.875rem !important;
    padding: 10px 14px !important;
    box-shadow: none !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.08) !important;
}

/* ── buttons ── */
.stButton > button {
    border-radius: 8px !important;
    border: 1px solid #e5e7eb !important;
    background: #ffffff !important;
    color: #374151 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 1rem !important;
    transition: background 0.15s;
}
.stButton > button:hover {
    background: #f9fafb !important;
    border-color: #d1d5db !important;
}

/* file uploader */
[data-testid="stFileUploader"] {
    border: 1px dashed #d1d5db;
    border-radius: 8px;
    padding: 10px;
    background: #fafafa;
}

/* divider */
hr { border: none; border-top: 1px solid #e5e7eb; margin: 1rem 0; }

/* expander */
[data-testid="stExpander"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
    background: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="page-title">✦ RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">LangGraph · Llama 3 · ChromaDB</div>', unsafe_allow_html=True)

    # ── API status ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">System status</div>', unsafe_allow_html=True)
    groq_ok   = bool(GROQ_API_KEY   and GROQ_API_KEY   != "your_groq_api_key_here")
    tavily_ok = bool(TAVILY_API_KEY and TAVILY_API_KEY != "your_tavily_api_key_here")
    cohere_ok = bool(COHERE_API_KEY and COHERE_API_KEY != "your_cohere_api_key_here")

    def _dot(ok): return f'<span class="dot {"dot-on" if ok else "dot-off"}"></span>'

    st.markdown(f"""
        <div class="status-row">{_dot(groq_ok)}   Groq — Llama 3 inference</div>
        <div class="status-row">{_dot(tavily_ok)} Tavily web search</div>
        <div class="status-row">{_dot(cohere_ok)} Cohere reranking</div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── pipeline trace (last run) ───────────────────────────────────────────
    st.markdown('<div class="section-label">Pipeline — last run</div>', unsafe_allow_html=True)

    pipeline_state = st.session_state.get("pipeline_state", {
        "planner":   "idle",
        "retrieval": "idle",
        "web":       "idle",
        "generate":  "idle",
    })

    def _step_html(label, icon, state):
        cls  = {"done": "step-done", "active": "step-active"}.get(state, "")
        mark = {"done": "✓", "active": "…"}.get(state, "·")
        return f'<div class="step {cls}"><span class="step-icon">{mark}</span>{label}</div>'

    st.markdown(
        _step_html("Query planner",    "🗂", pipeline_state["planner"])  +
        _step_html("Vector retrieval", "🔍", pipeline_state["retrieval"]) +
        _step_html("Web research",     "🌐", pipeline_state["web"])       +
        _step_html("Response synthesis","✍", pipeline_state["generate"]),
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── document upload ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Add documents</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("PDF, TXT, or MD", type=["pdf", "txt", "md"], label_visibility="collapsed")

    if uploaded:
        if "processed_files" not in st.session_state:
            st.session_state.processed_files = set()
        key = f"{uploaded.name}_{uploaded.size}"
        if key not in st.session_state.processed_files:
            with st.spinner(f"Ingesting {uploaded.name}…"):
                try:
                    tmp = DATA_RAW_DIR / uploaded.name
                    with open(tmp, "wb") as f:
                        f.write(uploaded.getbuffer())
                    result = get_ingestion_pipeline().ingest_file(tmp)
                    if tmp.exists():
                        tmp.unlink()
                    if result.success:
                        st.success(f"Added {result.chunks_ingested} chunks from {uploaded.name}")
                        st.session_state.processed_files.add(key)
                        st.rerun()
                    else:
                        st.error(f"Ingestion failed: {result.error}")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── corpus overview ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Indexed sources</div>', unsafe_allow_html=True)
    sources = set()
    chunk_count = 0
    try:
        db = get_vector_db()
        result = db.collection.get(include=["metadatas"])
        for m in result.get("metadatas", []):
            if m and "source" in m:
                sources.add(m["source"])
        chunk_count = len(result.get("ids", []))
    except Exception:
        pass

    if sources:
        for s in sorted(sources):
            icon = "🌐" if s.startswith("http") else "📄"
            st.markdown(f'<div class="source-chip">{icon} {Path(s).name}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:0.8rem;color:#9ca3af">No documents yet.</p>', unsafe_allow_html=True)

    st.markdown(
        f'<p style="font-size:0.78rem;color:#6b7280;margin-top:6px">'
        f'<b>{chunk_count}</b> chunks · <b>{len(sources)}</b> documents</p>',
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── danger zone ─────────────────────────────────────────────────────────
    if st.button("🗑 Clear database", use_container_width=True):
        try:
            db = get_vector_db()
            ids = db.collection.get().get("ids", [])
            if ids:
                db.collection.delete(ids=ids)
                st.session_state.pop("processed_files", None)
                st.success("Database cleared.")
                st.rerun()
            else:
                st.info("Already empty.")
        except Exception as e:
            st.error(str(e))

    st.markdown(
        '<p style="font-size:0.7rem;color:#d1d5db;margin-top:auto;padding-top:1rem">'
        'Agentic RAG v0.1 · LangGraph</p>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None

st.markdown('<div class="page-title">Ask your documents</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-sub">Queries are answered using your indexed documents, '
    'augmented with live web search when needed.</div>',
    unsafe_allow_html=True,
)

# ── evaluation metrics (shown after first answer) ───────────────────────────
def _score_class(v):
    if v is None: return "val-mid"
    return "val-good" if v >= 0.85 else ("val-mid" if v >= 0.6 else "val-bad")

def _score_str(v):
    return f"{v:.2f}" if v is not None else "—"

if st.session_state.last_metrics:
    m = st.session_state.last_metrics
    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-name">Faithfulness</div>
            <div class="metric-val {_score_class(m.get('faithfulness'))}">{_score_str(m.get('faithfulness'))}</div>
            <div class="metric-hint">Grounded in sources</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Context precision</div>
            <div class="metric-val {_score_class(m.get('context_precision'))}">{_score_str(m.get('context_precision'))}</div>
            <div class="metric-hint">Relevant chunks retrieved</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Context recall</div>
            <div class="metric-val {_score_class(m.get('context_recall'))}">{_score_str(m.get('context_recall'))}</div>
            <div class="metric-hint">Ground-truth coverage</div>
        </div>
        <div class="metric-card">
            <div class="metric-name">Answer relevancy</div>
            <div class="metric-val {_score_class(m.get('answer_relevancy'))}">{_score_str(m.get('answer_relevancy'))}</div>
            <div class="metric-hint">Response matches query</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── chat history ─────────────────────────────────────────────────────────────
def _render_messages():
    html = '<div class="chat-wrap">'
    for msg in st.session_state.messages:
        role  = msg["role"]          # "user" | "assistant"
        text  = msg["content"]
        meta  = msg.get("meta", "")
        srcs  = msg.get("sources", [])

        if role == "user":
            html += f"""
            <div class="msg-user">
                <div class="bubble bubble-user">{text}</div>
                <div class="msg-meta">{meta}</div>
            </div>"""
        else:
            src_tags = "".join(
                f'<span class="src-tag">{"🌐" if s.startswith("http") else "📄"} {Path(s).name}</span>'
                for s in srcs
            )
            html += f"""
            <div class="msg-bot">
                <div class="bubble bubble-bot">{text}</div>
                {'<div class="src-tags">' + src_tags + '</div>' if src_tags else ''}
                <div class="msg-meta">{meta}</div>
            </div>"""
    html += "</div>"
    return html

if st.session_state.messages:
    st.markdown(_render_messages(), unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#9ca3af">
        <div style="font-size:2rem;margin-bottom:0.5rem">✦</div>
        <p style="font-size:0.9rem">Ask a question to get started.<br>
        Upload documents in the sidebar to include them as context.</p>
    </div>
    """, unsafe_allow_html=True)

# ── input bar ────────────────────────────────────────────────────────────────
st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
query = st.text_input(
    "Query",
    placeholder="What would you like to know?",
    label_visibility="collapsed",
    key="query_input",
)

col_send, col_clear = st.columns([1, 5])
with col_send:
    send = st.button("Send →", use_container_width=True)
with col_clear:
    if st.button("Clear chat", use_container_width=False):
        st.session_state.messages = []
        st.session_state.last_metrics = None
        st.session_state.pipeline_state = {k: "idle" for k in ["planner","retrieval","web","generate"]}
        st.rerun()

# ── run graph on submit ───────────────────────────────────────────────────────
if send and query.strip():
    q = query.strip()

    # record user message
    ts = time.strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": q, "meta": f"You · {ts}"})

    # update pipeline state to active
    st.session_state.pipeline_state = {
        "planner":   "active",
        "retrieval": "idle",
        "web":       "idle",
        "generate":  "idle",
    }

    with st.spinner("Thinking…"):
        graph = get_rag_graph()
        t0 = time.time()

        # ── stage: planner ──
        st.session_state.pipeline_state["planner"]   = "done"
        st.session_state.pipeline_state["retrieval"] = "active"

        final_state = graph.execute_compiled_graph(q)
        elapsed = time.time() - t0

        # ── stage: all done ──
        st.session_state.pipeline_state = {
            "planner":   "done",
            "retrieval": "done",
            "web":       "done",
            "generate":  "done",
        }

    # collect sources
    vec_srcs = [
        c.get("source", "")
        for c in final_state.get("vector_context", [])
        if isinstance(c, dict) and c.get("source")
    ]
    web_srcs = [
        c.get("url", c.get("source", ""))
        for c in final_state.get("web_context", [])
        if isinstance(c, dict)
    ]
    all_srcs = list(dict.fromkeys(vec_srcs + web_srcs))  # deduplicated, ordered

    # store assistant message
    st.session_state.messages.append({
        "role":    "assistant",
        "content": final_state.get("response", "No response generated."),
        "meta":    f"RAG Assistant · {ts} · {elapsed:.2f}s · "
                   f"{len(vec_srcs)} doc chunk{'s' if len(vec_srcs) != 1 else ''} · "
                   f"{len(web_srcs)} web result{'s' if len(web_srcs) != 1 else ''}",
        "sources": all_srcs,
    })

   # ── evaluation metrics ──────────────────────────────────────────────────────
    eval_dataset = [{
        "question":     q,
        "answer":       final_state.get("response", ""),
        "contexts":     [c.get("content", str(c)) for c in final_state.get("vector_context", [])],
        "ground_truth": "",   # no ground truth at query time; RAGAS degrades gracefully
    }]

    try:
        evaluator = get_evaluator()
        eval_scores = evaluator.evaluate(eval_dataset)
    except Exception as e:
        eval_scores = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_recall": None,
            "context_precision": None,
        }

    st.session_state.last_metrics = {
        "faithfulness":      eval_scores.get("faithfulness"),
        "context_precision": eval_scores.get("context_precision"),
        "context_recall":    eval_scores.get("context_recall"),
        "answer_relevancy":  eval_scores.get("answer_relevancy"),
    }

    # ── optional expanders for raw context ──────────────────────────────────
    with st.expander("Retrieved context"):
        col_v, col_w = st.columns(2)
        with col_v:
            st.markdown(f"**ChromaDB — {len(final_state.get('vector_context', []))} chunks**")
            st.json(final_state.get("vector_context", []), expanded=False)
        with col_w:
            st.markdown(f"**Web — {len(final_state.get('web_context', []))} results**")
            st.json(final_state.get("web_context", []), expanded=False)

    with st.expander("Planner output"):
        st.json(final_state.get("plan", {}), expanded=False)

    st.rerun()