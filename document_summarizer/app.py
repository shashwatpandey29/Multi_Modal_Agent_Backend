# app.py
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Research Paper Analyzer",
    layout="wide",
    page_icon="📄"
)

# ---------- Custom Styling ----------
st.markdown("""
<style>
.main-title {
    font-size: 36px;
    font-weight: bold;
}
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 20px;
}
.metric-box {
    background-color: #f0f2f6;
    padding: 12px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📄 AI Research Paper Analyzer</div>', unsafe_allow_html=True)

# ============================================================
# -------------------- Upload Section ------------------------
# ============================================================

st.markdown('<div class="section-title">Upload Research Paper</div>', unsafe_allow_html=True)

if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

if "last_upload" not in st.session_state:
    st.session_state.last_upload = None

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "txt"]
)

if uploaded_file and not st.session_state.uploaded:
    with st.spinner("Uploading and analyzing paper..."):
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": (uploaded_file.name, uploaded_file.getvalue())},
        )

        if response.ok:
            data = response.json()
            st.session_state.uploaded = True
            st.session_state.last_upload = data
            st.success("Paper processed successfully!")
        else:
            st.error("Upload failed")

# Show metrics after upload
if st.session_state.last_upload:
    data = st.session_state.last_upload

    col1, col2, col3 = st.columns(3)
    col1.metric("Paper ID", data["paper_id"])
    col2.metric("Analysis Time (sec)", round(data["analysis_time_sec"], 2))
    col3.metric("Cached", data.get("cached", False))

# Optional reset button
if st.session_state.uploaded:
    if st.button("Upload Another Paper"):
        st.session_state.uploaded = False
        st.session_state.last_upload = None


# ============================================================
# -------------------- Paper Selection -----------------------
# ============================================================

st.markdown('<div class="section-title">Select Research Paper</div>', unsafe_allow_html=True)

papers_resp = requests.get(f"{API_URL}/papers")
papers = papers_resp.json() if papers_resp.ok else []

if not papers:
    st.info("No papers uploaded yet.")
    st.stop()

paper_map = {
    f"{p['paper_id']} — {p['filename']}": p["paper_id"]
    for p in papers
}

selected_label = st.selectbox("Choose a paper", paper_map.keys())
paper_id = paper_map[selected_label]

st.divider()

# ============================================================
# -------------------- Tabs Layout ---------------------------
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📘 Summary",
    "📊 Full Analysis",
    "📈 Stats",
    "🔍 Semantic Search",
    "💬 Chat"
])

# ============================================================
# -------------------- SUMMARY TAB ---------------------------
# ============================================================

with tab1:
    if st.button("Show Summary"):
        with st.spinner("Fetching summary..."):
            res = requests.get(f"{API_URL}/summary/{paper_id}")
            if res.ok:
                st.markdown(res.json()["summary"])
            else:
                st.error("Failed to fetch summary")

# ============================================================
# -------------------- FULL ANALYSIS TAB ---------------------
# ============================================================

with tab2:
    if st.button("Load Full Analysis"):
        with st.spinner("Fetching full analysis..."):
            res = requests.get(f"{API_URL}/analysis/{paper_id}")
            if res.ok:
                data = res.json()

                st.subheader("📘 Summary")
                st.markdown(data["summary"])

                st.subheader("🧠 Key Learnings")
                st.markdown(data["key_learnings"])

                st.subheader("🚀 Contributions")
                st.markdown(data["contributions"])

                st.subheader("⚠ Limitations")
                st.markdown(data["limitations"])

                st.caption(f"⏱ Analysis Time: {round(data['analysis_time_sec'],2)} seconds")

            else:
                st.error("Analysis not available")

# ============================================================
# -------------------- STATS TAB -----------------------------
# ============================================================

with tab3:
    res = requests.get(f"{API_URL}/stats/{paper_id}")
    if res.ok:
        stats = res.json()

        col1, col2 = st.columns(2)
        col1.metric("Total Chunks", stats["total_chunks"])
        col2.metric("Total Questions Asked", stats["total_questions"])
    else:
        st.error("Failed to fetch stats")

# ============================================================
# -------------------- SEMANTIC SEARCH TAB -------------------
# ============================================================

with tab4:
    query = st.text_input("Search semantically inside paper")

    if st.button("Search") and query:
        with st.spinner("Searching..."):
            payload = {"paper_id": paper_id, "query": query}
            res = requests.post(f"{API_URL}/search", json=payload)

            if res.ok:
                results = res.json()["results"]
                if results:
                    for i, r in enumerate(results):
                        st.markdown(f"**Result {i+1}:**")
                        st.markdown(r)
                        st.markdown("---")
                else:
                    st.info("No relevant sections found.")
            else:
                st.error("Search failed")

# ============================================================
# -------------------- CHAT TAB ------------------------------
# ============================================================

with tab5:

    if "chat" not in st.session_state:
        st.session_state.chat = []

    question = st.text_input("Ask a question about the paper")

    if st.button("Ask") and question:
        with st.spinner("Thinking..."):
            payload = {
                "paper_id": paper_id,
                "question": question
            }
            res = requests.post(f"{API_URL}/ask", json=payload)

            if res.ok:
                data = res.json()
                st.session_state.chat.append(
                    (question, data["answer"], data["response_time_sec"], data.get("cached", False))
                )
            else:
                st.error("Error while asking question")

    for q, a, t, cached in st.session_state.chat[::-1]:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**AI:** {a}")
        st.caption(f"⏱ {round(t,2)}s | Cached: {cached}")
        st.markdown("---")
