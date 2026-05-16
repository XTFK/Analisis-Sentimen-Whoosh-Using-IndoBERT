import streamlit as st
import torch
import re
import string
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Whoosh Sentiment Analyzer",
    page_icon="🚄",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
# Ganti dengan path model kamu di Hugging Face Hub, contoh:
# "username/whoosh-indobert-sentiment"
MODEL_NAME = "xtfk/Whoosh_IndoBERT_Sentiment"   # ← GANTI INI setelah upload ke HF Hub

LABEL_MAP  = {0: "Negatif", 1: "Netral", 2: "Positif"}
LABEL_EMOJI = {0: "😠", 1: "😐", 2: "😊"}
LABEL_COLOR = {0: "#e74c3c", 1: "#3498db", 2: "#2ecc71"}

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.main { background: #0d0d0d; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #e94560;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero h1 { font-size: 2rem; font-weight: 700; color: #fff; margin: 0 0 .4rem; }
.hero p  { color: #aaa; margin: 0; font-size: .95rem; }
.hero span { color: #e94560; }

.result-box {
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-top: 1.5rem;
    text-align: center;
    border: 2px solid;
}
.result-label { font-size: 2rem; font-weight: 700; margin-bottom: .3rem; }
.result-conf  { font-size: .9rem; color: #ccc; font-family: 'DM Mono', monospace; }

.prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: .4rem 0;
    font-size: .88rem;
}
.prob-bar-bg {
    flex: 1;
    height: 8px;
    background: #222;
    border-radius: 4px;
    margin: 0 .8rem;
    overflow: hidden;
}
.prob-bar { height: 100%; border-radius: 4px; transition: width .6s ease; }

.info-box {
    background: #111;
    border: 1px solid #333;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: .82rem;
    color: #888;
    margin-top: 1rem;
    font-family: 'DM Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚄 Whoosh <span>Sentiment</span> Analyzer</h1>
  <p>Analisis sentimen komentar TikTok tentang Kereta Cepat Whoosh<br>
  menggunakan model <b>IndoBERT</b> fine-tuned Bahasa Indonesia</p>
</div>
""", unsafe_allow_html=True)

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

with st.spinner("⏳ Memuat model IndoBERT... (hanya sekali, harap tunggu)"):
    tokenizer, model = load_model()

st.success("✅ Model siap digunakan!")

# ── Text cleaning (sama dengan notebook) ─────────────────────────────────────
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

# ── Predict function ──────────────────────────────────────────────────────────
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

# ── Input area ────────────────────────────────────────────────────────────────
st.markdown("### ✍️ Masukkan Komentar")
user_input = st.text_area(
    label="",
    placeholder="Contoh: Whoosh cepet banget, worth it banget buat perjalanan Bandung-Jakarta!",
    height=130,
    label_visibility="collapsed",
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_btn = st.button("🔍 Analisis Sentimen", use_container_width=True, type="primary")

# ── Contoh komentar ───────────────────────────────────────────────────────────
with st.expander("💡 Coba contoh komentar"):
    examples = [
        "Whoosh kerennn, Jakarta Bandung cuma 40 menit gila sih!",
        "Tiketnya mahal banget, gak worth it sama sekali",
        "Udah nyoba kemarin, kursinya nyaman dan tepat waktu",
        "Biasa aja sih, sama kayak kereta biasa tapi lebih mahal",
        "Pelayanannya buruk banget, AC rusak dan petugasnya gak helpful",
    ]
    for ex in examples:
        if st.button(f"📌 {ex[:60]}...", key=ex):
            user_input = ex
            analyze_btn = True

# ── Result ────────────────────────────────────────────────────────────────────
if analyze_btn and user_input.strip():
    with st.spinner("🔄 Menganalisis..."):
        label_id, probs, cleaned = predict(user_input)

    label = LABEL_MAP[label_id]
    color = LABEL_COLOR[label_id]
    emj   = LABEL_EMOJI[label_id]
    conf  = probs[label_id] * 100

    # Result box
    st.markdown(f"""
    <div class="result-box" style="border-color:{color}; background:{color}18;">
        <div class="result-label" style="color:{color};">{emj} {label}</div>
        <div class="result-conf">Confidence: {conf:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars
    st.markdown("#### 📊 Distribusi Probabilitas")
    for i, (lbl, clr) in enumerate(zip(LABEL_MAP.values(), LABEL_COLOR.values())):
        pct = probs[i] * 100
        st.markdown(f"""
        <div class="prob-row">
            <span style="width:70px; color:{clr};">{LABEL_EMOJI[i]} {lbl}</span>
            <div class="prob-bar-bg">
                <div class="prob-bar" style="width:{pct:.1f}%; background:{clr};"></div>
            </div>
            <span style="width:50px; text-align:right; font-family:'DM Mono',monospace; color:#ccc;">
                {pct:.1f}%
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Debug info
    with st.expander("🔧 Detail preprocessing"):
        st.markdown(f"""
        <div class="info-box">
        <b>Input asli:</b><br>{user_input}<br><br>
        <b>Setelah cleaning:</b><br>{cleaned}
        </div>
        """, unsafe_allow_html=True)

elif analyze_btn and not user_input.strip():
    st.warning("⚠️ Mohon masukkan komentar terlebih dahulu.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:.8rem;'>"
    "Model: IndoBERT fine-tuned · Dataset: Komentar TikTok Whoosh · "
    "Built with Streamlit 🎈</p>",
    unsafe_allow_html=True,
)
