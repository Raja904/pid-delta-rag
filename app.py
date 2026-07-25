"""
app.py
Streamlit web UI for delta-chat.
Run: streamlit run app.py
"""
import json, os, sys, time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "delta-chat | Document Delta & Chat",
    page_icon   = "📄",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0f1115; }
  .block-container { max-width: 1100px; }
  .stButton button {
    background: linear-gradient(90deg,#4c8dff,#7c5cff);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 0.5rem 1.2rem;
  }
  .citation-box {
    background: #161a21; border-left: 3px solid #4c8dff;
    border-radius: 6px; padding: 10px 14px; margin: 6px 0;
    font-size: 0.82rem; color: #9aa7b8;
  }
  .delta-added   { border-left-color: #2fbf71 !important; }
  .delta-removed { border-left-color: #ef5d5d !important; }
  .delta-modified{ border-left-color: #f5a524 !important; }
  .metric-card {
    background: #161a21; border: 1px solid #2a323e;
    border-radius: 12px; padding: 14px 18px; text-align: center;
  }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 delta-chat")
    st.caption("Document Delta & Grounded Chat")
    st.divider()

    groq_key = st.text_input(
        "Groq API Key", type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get a free key at console.groq.com",
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    st.divider()
    st.markdown("### Upload Documents")
    file_a = st.file_uploader("Document A (base revision)", type=["pdf"])
    file_b = st.file_uploader("Document B (revised)", type=["pdf"])
    scanned_b = st.checkbox("Document B is scanned (use OCR)", value=False)
    rev_a = st.text_input("Revision label A", value="A")
    rev_b = st.text_input("Revision label B", value="B")

    run_btn = st.button("⚡ Run Delta Analysis", use_container_width=True,
                        disabled=(file_a is None or file_b is None))

    st.divider()
    st.caption("Formats: Native PDF · Scanned PDF · DWG (stub)")

# ── Session state ─────────────────────────────────────────────────────────────
if "delta_items" not in st.session_state:
    st.session_state.delta_items = []
if "doc_a" not in st.session_state:
    st.session_state.doc_a = None
if "doc_b" not in st.session_state:
    st.session_state.doc_b = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "report_json" not in st.session_state:
    st.session_state.report_json = None

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn and file_a and file_b:
    from src.ingest.base import AdapterRegistry
    from src.ingest.pdf_native import NativePDFAdapter
    from src.ingest.pdf_scanned import ScannedPDFAdapter
    from src.ingest.dwg import DWGAdapter
    from src.delta import engine as delta_engine
    from src.delta.report import render as render_report
    from src.chat.index import build_index
    from src.observability.tracer import start_trace, finish_trace, trace_stage
    from src.observability.logger import new_request_id

    # Save uploads to temp files
    tmp_dir = Path("data/samples/session")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    path_a = tmp_dir / file_a.name
    path_b = tmp_dir / file_b.name
    path_a.write_bytes(file_a.read())
    path_b.write_bytes(file_b.read())

    rid = new_request_id()
    start_trace("ingest_and_delta", {"pid_a": file_a.name, "pid_b": file_b.name})

    with st.status("⚙️ Running pipeline...", expanded=True) as status:
        registry = AdapterRegistry()
        registry.register(NativePDFAdapter())
        registry.register(DWGAdapter())

        st.write("📥 Ingesting Document A...")
        with trace_stage("ingest_a"):
            doc_a = registry.ingest(path_a, pid=path_a.stem, revision=rev_a)
        st.write(f"✅ Doc A: {len(doc_a.pages)} pages, {len(doc_a.all_blocks())} blocks")

        st.write("📥 Ingesting Document B...")
        with trace_stage("ingest_b"):
            if scanned_b:
                doc_b = ScannedPDFAdapter().ingest(path_b, pid=path_b.stem, revision=rev_b)
            else:
                doc_b = registry.ingest(path_b, pid=path_b.stem, revision=rev_b)
        st.write(f"✅ Doc B: {len(doc_b.pages)} pages, {len(doc_b.all_blocks())} blocks")

        st.write("🔍 Computing delta...")
        with trace_stage("delta"):
            items = delta_engine.run(doc_a, doc_b)
        st.write(f"✅ {len(items)} changes detected")

        st.write("📝 Generating report...")
        with trace_stage("report"):
            md_path, json_path = render_report(items, doc_a, doc_b, run_id=rid)

        st.write("🗂️ Building chat index...")
        with trace_stage("index"):
            build_index(doc_a, doc_b, items)

        trace_path = finish_trace()
        status.update(label="✅ Pipeline complete!", state="complete")

    st.session_state.delta_items = items
    st.session_state.doc_a       = doc_a
    st.session_state.doc_b       = doc_b
    st.session_state.indexed     = True
    st.session_state.report_json = json_path
    st.session_state.chat_history = []
    st.rerun()

# ── Main content ──────────────────────────────────────────────────────────────
app_mode = st.radio(
    "Navigation", 
    ["📊 Delta Report", "💬 Grounded Chat", "🔍 Trace Viewer"], 
    horizontal=True,
    label_visibility="collapsed"
)

# ── Mode 1: Delta Report ───────────────────────────────────────────────────────
if app_mode == "📊 Delta Report":
    items = st.session_state.delta_items
    doc_a = st.session_state.doc_a
    doc_b = st.session_state.doc_b

    if not items and not doc_a:
        st.info("👈 Upload two documents in the sidebar and click **Run Delta Analysis** to start.")
        st.stop()

    if doc_a and doc_b:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""<div class="metric-card">
                <div style="color:#9aa7b8;font-size:12px">TOTAL CHANGES</div>
                <div style="font-size:28px;font-weight:700">{len(items)}</div></div>""",
                unsafe_allow_html=True)
        with col2:
            n_add = sum(1 for i in items if i.change_type=="added")
            st.markdown(f"""<div class="metric-card">
                <div style="color:#2fbf71;font-size:12px">ADDED</div>
                <div style="font-size:28px;font-weight:700;color:#2fbf71">{n_add}</div></div>""",
                unsafe_allow_html=True)
        with col3:
            n_rem = sum(1 for i in items if i.change_type=="removed")
            st.markdown(f"""<div class="metric-card">
                <div style="color:#ef5d5d;font-size:12px">REMOVED</div>
                <div style="font-size:28px;font-weight:700;color:#ef5d5d">{n_rem}</div></div>""",
                unsafe_allow_html=True)
        with col4:
            n_mod = sum(1 for i in items if i.change_type=="modified")
            st.markdown(f"""<div class="metric-card">
                <div style="color:#f5a524;font-size:12px">MODIFIED</div>
                <div style="font-size:28px;font-weight:700;color:#f5a524">{n_mod}</div></div>""",
                unsafe_allow_html=True)

        st.markdown("---")

    if not items:
        st.success("No meaningful differences detected between the documents.")
    else:
        # Filters
        ftype = st.multiselect(
            "Filter by change type",
            ["added","removed","modified"],
            default=["added","removed","modified"],
        )
        filtered = [i for i in items if i.change_type in ftype]
        st.caption(f"Showing {len(filtered)} of {len(items)} changes")

        color_map = {"added":"#2fbf71","removed":"#ef5d5d","modified":"#f5a524"}
        for item in filtered:
            color = color_map.get(item.change_type, "#4c8dff")
            page_label = f"p.{item.page_b or item.page_a}"
            with st.expander(
                f"**{item.change_type.upper()}** · {item.content_type} · {page_label} · "
                f"`{item.change_id}` · conf: {item.confidence:.0%}",
                expanded=False,
            ):
                c1, c2 = st.columns(2)
                with c1:
                    if item.old_value:
                        st.markdown("**Before (Doc A)**")
                        st.code(item.old_value[:500], language=None)
                with c2:
                    if item.new_value:
                        st.markdown("**After (Doc B)**")
                        st.code(item.new_value[:500], language=None)
                st.caption(item.description)

    # Download report
    if st.session_state.report_json:
        json_path = Path(st.session_state.report_json)
        if json_path.exists():
            st.download_button(
                "⬇️ Download JSON Report",
                data=json_path.read_text(encoding="utf-8"),
                file_name=json_path.name,
                mime="application/json",
            )

# ── Mode 2: Grounded Chat ────────────────────────────────────────────────────────
elif app_mode == "💬 Grounded Chat":
    if not st.session_state.indexed:
        st.info("👈 Run the delta analysis first to enable grounded chat.")
    else:
        st.markdown("### 💬 Ask questions about the documents or the delta")
        st.caption("Answers are grounded with citations to source content.")

        # Chat history display
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("citations"):
                    with st.expander("📎 Citations"):
                        for cit in msg["citations"]:
                            st.markdown(
                                f'<div class="citation-box"><b>[{cit["ref"]}]</b> {cit["text"][:150]}...</div>',
                                unsafe_allow_html=True,
                            )

        if prompt := st.chat_input("Ask about the documents or changes..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    from src.chat.answer import answer as get_answer
                    from src.observability.tracer import start_trace, finish_trace
                    from src.observability.logger import new_request_id

                    try:
                        new_request_id()
                        start_trace("chat", {"question": prompt[:80]})
                        
                        summary_dict = None
                        if st.session_state.report_json:
                            try:
                                with open(st.session_state.report_json, 'r', encoding='utf-8') as f:
                                    report_data = json.load(f)
                                    summary_dict = report_data.get("summary")
                            except Exception:
                                pass
                                
                        result = get_answer(prompt, summary_dict=summary_dict)
                        finish_trace()
                        st.markdown(result.answer)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result.answer,
                            "citations": [
                                {"ref": c.ref, "text": c.text}
                                for c in result.citations
                            ],
                        })
                        if result.citations:
                            with st.expander("📎 Citations"):
                                for cit in result.citations:
                                    st.markdown(
                                        f'<div class="citation-box"><b>[{cit.ref}]</b> {cit.text[:150]}...</div>',
                                        unsafe_allow_html=True,
                                    )
                    except Exception as e:
                        st.error(f"Error: {e}")
                        finish_trace(error=str(e))

# ── Mode 3: Trace Viewer ────────────────────────────────────────────────────────
elif app_mode == "🔍 Trace Viewer":
    st.markdown("### 🔍 Request Traces")
    traces_dir = Path("traces")
    trace_files = sorted(traces_dir.glob("*.json"), reverse=True) if traces_dir.exists() else []

    if not trace_files:
        st.info("No traces yet. Run an analysis or chat query first.")
    else:
        selected = st.selectbox("Select trace", [f.name for f in trace_files])
        trace_data = json.loads((traces_dir / selected).read_text())

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Latency", f"{trace_data.get('total_latency_s',0):.2f}s")
        col2.metric("Total Cost", f"${trace_data.get('total_cost_usd',0):.6f}")
        col3.metric("Total Tokens", trace_data.get("total_tokens", 0))

        st.markdown("#### Pipeline Stages")
        for stage in trace_data.get("stages", []):
            color = "🔴" if stage.get("error") else "🟢"
            st.markdown(f"{color} **{stage['name']}** — {stage['latency_s']:.3f}s "
                        + (f"| ❌ {stage['error']}" if stage.get('error') else ""))

        if trace_data.get("llm_calls"):
            st.markdown("#### LLM Calls")
            for call in trace_data["llm_calls"]:
                with st.expander(f"{call['model']} — {call['latency_s']:.2f}s — ${call['cost_usd']:.6f}"):
                    st.markdown("**Prompt preview:**")
                    st.code(call["prompt_preview"], language=None)
                    st.markdown("**Response preview:**")
                    st.code(call["response_preview"], language=None)
                    st.caption(f"Tokens: {call['prompt_tokens']} in / {call['completion_tokens']} out")

        with st.expander("Raw JSON"):
            st.json(trace_data)
