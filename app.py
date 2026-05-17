import streamlit as st
import torch
import re
import string
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Whoosh Sentiment Analyzer",
    page_icon="",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME = "xtfk/Whoosh_IndoBERT_Sentiment"   # ← Ganti setelah upload ke HF Hub

LABEL_MAP   = {0: "Negatif", 1: "Netral", 2: "Positif"}
LABEL_COLOR = {0: "#C0392B", 1: "#2C3E50", 2: "#1A6B3C"}

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #ffffff;
    color: #1a1a1a;
}

.stApp { background-color: #ffffff; }
.block-container { padding-top: 3rem; max-width: 720px; }

/* Header */
.header {
    border-bottom: 1px solid #e8e8e8;
    padding-bottom: 2rem;
    margin-bottom: 2.5rem;
}
.header-label {
    font-size: .7rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: #999;
    font-weight: 500;
    margin-bottom: .6rem;
}
.header h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: #111;
    margin: 0 0 .5rem;
    line-height: 1.15;
}
.header p {
    font-size: .875rem;
    color: #777;
    margin: 0;
    line-height: 1.6;
    font-weight: 300;
}

/* Section label */
.section-label {
    font-size: .7rem;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: #aaa;
    font-weight: 500;
    margin-bottom: .75rem;
}

/* Result card */
.result-card {
    border: 1px solid #e8e8e8;
    border-radius: 4px;
    padding: 2rem 2.5rem;
    margin-top: 1.5rem;
    margin-bottom: 2rem;
}
.result-sentiment {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.2rem;
    font-weight: 600;
    letter-spacing: .02em;
    margin-bottom: .25rem;
}
.result-conf {
    font-size: .78rem;
    color: #999;
    letter-spacing: .04em;
    font-weight: 400;
}

/* Probability bars */
.prob-section { margin-top: 1.5rem; }
.prob-row {
    display: flex;
    align-items: center;
    margin: .75rem 0;
    gap: 1rem;
}
.prob-label {
    width: 64px;
    font-size: .78rem;
    color: #555;
    letter-spacing: .03em;
    font-weight: 500;
    flex-shrink: 0;
}
.prob-bar-bg {
    flex: 1;
    height: 2px;
    background: #ebebeb;
    border-radius: 2px;
    overflow: hidden;
}
.prob-bar {
    height: 100%;
    border-radius: 2px;
}
.prob-pct {
    width: 44px;
    text-align: right;
    font-size: .78rem;
    color: #999;
    font-weight: 400;
    flex-shrink: 0;
    letter-spacing: .02em;
}

/* Divider */
.thin-divider {
    border: none;
    border-top: 1px solid #f0f0f0;
    margin: 1.5rem 0;
}

/* Info box */
.info-box {
    background: #fafafa;
    border: 1px solid #ebebeb;
    border-radius: 4px;
    padding: 1.2rem 1.5rem;
    font-size: .78rem;
    color: #888;
    line-height: 1.7;
}
.info-box b { color: #555; font-weight: 500; }

/* Streamlit widget overrides */
.stTextArea textarea {
    border: 1px solid #e0e0e0 !important;
    border-radius: 4px !important;
    font-size: .875rem !important;
    color: #1a1a1a !important;
    background: #fff !important;
    padding: 1rem !important;
    font-family: 'Inter', sans-serif !important;
    resize: none !important;
    box-shadow: none !important;
}
.stTextArea textarea:focus {
    border-color: #999 !important;
    box-shadow: none !important;
}
.stButton button {
    background: #111 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 4px !important;
    font-size: .72rem !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
    padding: .7rem 2rem !important;
    transition: background .2s ease !important;
}
.stButton button:hover { background: #333 !important; }

/* Footer */
.footer {
    border-top: 1px solid #f0f0f0;
    padding-top: 1.5rem;
    margin-top: 3rem;
    font-size: .72rem;
    color: #ccc;
    letter-spacing: .06em;
    text-align: center;
}

/* Expander */
.streamlit-expanderHeader {
    font-size: .72rem !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    color: #aaa !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
  <div class="header-label">Analisis Teks Berbahasa Indonesia</div>
  <h1>Whoosh<br>Sentiment Analyzer</h1>
  <p>
    Sistem klasifikasi sentimen komentar publik terhadap Kereta Cepat Whoosh
    menggunakan model IndoBERT yang telah dilakukan fine-tuning pada data
    komentar TikTok berbahasa Indonesia.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

with st.spinner("Memuat model IndoBERT..."):
    tokenizer, model = load_model()

st.success("Model berhasil dimuat dan siap digunakan.")

# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = emoji.replace_emoji(text, replace='')
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Predict ───────────────────────────────────────────────────────────────────
def predict(text: str):
    cleaned = clean_text(text)
    inputs  = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    probs      = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_label = int(torch.argmax(logits, dim=-1).item())
    return pred_label, probs, cleaned

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Masukkan Komentar</div>', unsafe_allow_html=True)
user_input = st.text_area(
    label="",
    placeholder="Tuliskan komentar yang ingin dianalisis...",
    height=120,
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([2, 2, 2])
with col2:
    analyze_btn = st.button("Analisis", use_container_width=True)

# ── Contoh komentar ───────────────────────────────────────────────────────────
with st.expander("Contoh Komentar"):
    examples = [
        "Whoosh sangat cepat, perjalanan Jakarta–Bandung hanya 40 menit.",
        "Harga tiket terlalu mahal, tidak sebanding dengan fasilitasnya.",
        "Sudah mencoba kemarin, kursi nyaman dan keberangkatan tepat waktu.",
        "Biasa saja, tidak jauh berbeda dengan kereta reguler namun lebih mahal.",
        "Pelayanan kurang memuaskan, AC tidak berfungsi optimal.",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            user_input  = ex
            analyze_btn = True

# ── Result ────────────────────────────────────────────────────────────────────
if analyze_btn and user_input.strip():
    with st.spinner("Menganalisis komentar..."):
        label_id, probs, cleaned = predict(user_input)

    label = LABEL_MAP[label_id]
    color = LABEL_COLOR[label_id]
    conf  = probs[label_id] * 100

    prob_bars_html = ""
    for i in range(3):
        lbl = LABEL_MAP[i]
        clr = LABEL_COLOR[i]
        pct = probs[i] * 100
        prob_bars_html += (
            f'<div class="prob-row">'
            f'<span class="prob-label">{lbl}</span>'
            f'<div class="prob-bar-bg">'
            f'<div class="prob-bar" style="width:{pct:.1f}%; background:{clr};"></div>'
            f'</div>'
            f'<span class="prob-pct">{pct:.1f}%</span>'
            f'</div>'
        )

    result_html = (
        f'<div class="result-card">'
        f'<div class="section-label">Hasil Analisis</div>'
        f'<div class="result-sentiment" style="color:{color};">{label}</div>'
        f'<div class="result-conf">Tingkat kepercayaan: {conf:.1f}%</div>'
        f'<hr class="thin-divider">'
        f'<div class="section-label">Distribusi Probabilitas</div>'
        f'<div class="prob-section">{prob_bars_html}</div>'
        f'</div>'
    )
    st.markdown(result_html, unsafe_allow_html=True)

    with st.expander("Detail Preprocessing"):
        st.markdown(f"""
        <div class="info-box">
            <b>Teks asli</b><br>{user_input}
            <br><br>
            <b>Setelah preprocessing</b><br>{cleaned}
        </div>
        """, unsafe_allow_html=True)

elif analyze_btn and not user_input.strip():
    st.warning("Mohon masukkan teks komentar terlebih dahulu.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Model IndoBERT Fine-tuned &nbsp;&middot;&nbsp;
    Dataset Komentar TikTok Whoosh &nbsp;&middot;&nbsp;
    Dibangun menggunakan Streamlit
</div>
""", unsafe_allow_html=True)
