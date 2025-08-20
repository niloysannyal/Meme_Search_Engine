# app.py

import streamlit as st
from pathlib import Path
from PIL import Image
from src.inference import MemeSearch
from src.config import DATA_DIR, LABEL_CSV, MODEL_NAME, SENTIMENT_OPTIONS
import base64

# ---------------------- Page Config -------------------------
st.set_page_config(page_title="Meme Search Engine", page_icon="🎭", layout="wide")
st.markdown("<h1 style='text-align: center;'>🔍 Meme Search Engine</h1>", unsafe_allow_html=True)

# ---------------------- Sidebar -------------------------
st.sidebar.markdown(
    "<h3 style='margin-bottom: 10px;'>🎛️ Search Controls</h3>",
    unsafe_allow_html=True
)

# Text Query Input
st.sidebar.markdown("**📝 Query:**", unsafe_allow_html=True)
query = st.sidebar.text_input("Search Query", "Exam", label_visibility="collapsed")

# Sentiment Filter
st.sidebar.markdown("**💬 Sentiment:**", unsafe_allow_html=True)
sentiment = st.sidebar.selectbox("Sentiment Filter", ["Any", "Positive", "Neutral", "Negative"], label_visibility="collapsed")

# Top K Slider
st.sidebar.markdown("**🔢 Top Results:**", unsafe_allow_html=True)
top_k = st.sidebar.slider("Top Results", 1, 10, 5, label_visibility="collapsed")

# Upload Box
st.sidebar.markdown("**📤 Upload an image:**", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Drop or browse", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

# Meme count display
st.sidebar.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

# ---------------------- Load meme paths -------------------------
DATA_DIR.mkdir(exist_ok=True)
meme_paths = list(DATA_DIR.glob("*.jpg")) + list(DATA_DIR.glob("*.jpeg")) + list(DATA_DIR.glob("*.png")) + list(DATA_DIR.glob("*.webp"))

st.sidebar.success(f"📁 {len(meme_paths)} memes loaded")

# ---------------------- Load Model -------------------------
@st.cache_resource
def load_search_engine():
    return MemeSearch(
        meme_paths=meme_paths,
        model_name=MODEL_NAME,
        label_csv=LABEL_CSV,
        embedding_path="Memes/embeddings.pkl"
    )

search_engine = load_search_engine()


# ---------------------- Search -------------------------
if uploaded_file:
    uploaded_image = Image.open(uploaded_file).convert("RGB")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    with st.spinner("Searching by uploaded image..."):
        results = search_engine.search_by_image(uploaded_image, top_k)
else:
    with st.spinner("Searching memes..."):
        results = search_engine.search(query, top_k, sentiment_filter=sentiment)

# ---------------------- Results -------------------------
st.markdown("---")
if uploaded_file:
    st.markdown("### 🎯 Similar Results for Uploaded Image")
else:
    sentiment_display = f"(Sentiment: {sentiment})" if sentiment != "Any" else ""
    st.markdown(f"### 🎯 Results for: **'{query}'** {sentiment_display}")

cols = st.columns(top_k)
for idx, (name, score) in enumerate(results):
    meme_path = DATA_DIR / name
    with cols[idx]:
        st.image(meme_path, caption=f"{name} ({score:.2%})", use_container_width=True)

        # Download button
        with open(meme_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            href = f'<a href="data:file/png;base64,{b64}" download="{name}">📥 Download</a>'
            st.markdown(href, unsafe_allow_html=True)

# ---------------------- Footer -------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-size: 14px;'>
  Made with ❤️ by <b>Niloy Sannyal</b> | <a href='https://github.com/NiloySannyal' target='_blank'>GitHub</a>
</div>
""", unsafe_allow_html=True)
