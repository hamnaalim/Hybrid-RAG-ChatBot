"""Streamlit frontend for Hybrid RAG."""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

# Allow running as `streamlit run frontend/streamlit_app.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Hybrid RAG",
    page_icon="📄",
    layout="wide",
)

st.title("Hybrid RAG Chatbot")
st.caption(
    "Dense + BM25 retrieval → cross-encoder rerank → grounded generation → citation verification"
)

with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=DEFAULT_API).rstrip("/")
    top_k = st.slider("Rerank top-k", min_value=1, max_value=10, value=5)
    skip_verification = st.checkbox("Skip citation verification", value=False)

    st.divider()
    st.subheader("Index status")
    if st.button("Refresh health"):
        st.session_state.pop("health", None)
    try:
        health = requests.get(f"{api_url}/health", timeout=10).json()
        st.session_state["health"] = health
    except requests.RequestException as exc:
        health = st.session_state.get("health")
        st.error(f"API unreachable: {exc}")
        health = None

    if health:
        st.write(f"Ready: **{health.get('index_ready')}**")
        st.write(f"Chunks: **{health.get('chunk_count', 0)}**")

    st.divider()
    st.subheader("Ingest documents")
    uploads = st.file_uploader(
        "PDF / DOCX / TXT / HTML",
        type=["pdf", "docx", "txt", "html", "htm", "md"],
        accept_multiple_files=True,
    )
    if st.button("Upload & index", disabled=not uploads):
        files = [("files", (f.name, f.getvalue(), f.type or "application/octet-stream")) for f in uploads]
        with st.spinner("Ingesting…"):
            try:
                resp = requests.post(f"{api_url}/ingest", files=files, timeout=600)
                resp.raise_for_status()
                data = resp.json()
                st.success(
                    f"Indexed {data['documents']} document parts → {data['chunks']} chunks"
                )
                st.write("Sources:", ", ".join(data.get("sources", [])))
            except requests.RequestException as exc:
                st.error(f"Ingest failed: {exc}")

    if st.button("Re-index data/raw_docs"):
        with st.spinner("Building index from disk…"):
            try:
                resp = requests.post(f"{api_url}/ingest/directory", timeout=600)
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Indexed {data['chunks']} chunks from disk")
            except requests.RequestException as exc:
                st.error(f"Directory ingest failed: {exc}")

question = st.text_input(
    "Ask a question about your documents",
    placeholder="How many annual leave days can employees carry forward?",
)

col_ask, col_clear = st.columns([1, 1])
with col_ask:
    ask = st.button("Ask", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear history", use_container_width=True):
        st.session_state.pop("history", None)

if "history" not in st.session_state:
    st.session_state.history = []

if ask and question.strip():
    with st.spinner("Retrieving, reranking, generating, verifying…"):
        try:
            resp = requests.post(
                f"{api_url}/query",
                json={
                    "question": question.strip(),
                    "top_k": top_k,
                    "skip_verification": skip_verification,
                },
                timeout=300,
            )
            resp.raise_for_status()
            result = resp.json()
            st.session_state.history.insert(0, result)
        except requests.RequestException as exc:
            detail = ""
            if getattr(exc, "response", None) is not None:
                try:
                    detail = exc.response.json().get("detail", "")
                except Exception:  # noqa: BLE001
                    detail = exc.response.text
            st.error(f"Query failed: {detail or exc}")

for item in st.session_state.history:
    st.markdown("---")
    st.markdown(f"**Q:** {item['question']}")
    st.markdown(f"**A:** {item['answer']}")

    if item.get("verified"):
        st.success("Citations verified")
    elif item.get("citations"):
        st.warning("Some citations present (verification may be partial)")
    else:
        st.info("No verified citations")

    if item.get("citations"):
        st.subheader("Citations")
        for cite in item["citations"]:
            badge = "✓" if cite.get("verified") else "?"
            st.markdown(
                f"- {badge} **{cite['source']}** (page {cite['page']}) — "
                f"score={cite.get('verification_score')}"
            )
            with st.expander(f"Passage [{cite['marker']}]"):
                st.write(cite.get("passage", ""))

    with st.expander("Retrieval details"):
        st.write("Reranked chunks:")
        for scored in item.get("reranked", []):
            st.markdown(
                f"**{scored['source']}** p.{scored['page']} · score={scored.get('score'):.4f}"
            )
            st.write(scored["content"][:500] + ("…" if len(scored["content"]) > 500 else ""))
        st.json(item.get("meta", {}))
