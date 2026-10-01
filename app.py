"""Streamlit UI. Run: streamlit run app.py"""
import streamlit as st

import config
from rag import RAG, rank_info, source_label

st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Document Q&A")


@st.cache_resource
def load_rag(collection: str, top_k: int, retrieval: str) -> RAG:
    return RAG(collection=collection, top_k=top_k, retrieval=retrieval)


collections = RAG.available_collections()
if not collections:
    st.error("No index found. Run `python ingest.py` first.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    collection = st.selectbox("Index", collections)
    top_k = st.slider("Chunks retrieved (top-k)", 1, 10, config.TOP_K)
    modes = config.RETRIEVAL_METHODS
    retrieval = st.radio("Search method", modes, index=modes.index(config.RETRIEVAL),
                         help="rerank = hybrid candidates reordered by a reranker model; "
                              "hybrid = meaning-based + keyword search combined")
    st.caption(f"LLM: `{config.LLM_MODEL}`  \nEmbeddings: `{config.EMBED_MODEL}`")
    if st.button("Clear chat"):
        st.session_state.history = []

rag = load_rag(collection, top_k, retrieval)
st.session_state.setdefault("history", [])


def render(result: dict):
    st.markdown(result["answer"])
    st.caption(
        f"⏱ {result['total_s']}s (retrieval {result['retrieve_s']}s) · "
        f"{result['prompt_tokens']} prompt tokens"
    )
    with st.expander("Sources"):
        for i, s in enumerate(result["sources"], 1):
            st.markdown(f"**[{i}] {source_label(s)}** · {rank_info(s)}")
            st.text(s["text"][:1200] + ("…" if len(s["text"]) > 1200 else ""))


for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        render(turn)

if q := st.chat_input("Ask a question about the documents"):
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking…"):
            result = rag.answer(q)
        render(result)
    st.session_state.history.append(result)
