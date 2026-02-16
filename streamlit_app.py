import streamlit as st
from transformers import pipeline
from PIL import Image
import cv2
import tempfile
import os

# ၁။ Dashboard UI အလှဆင်ခြင်း
st.set_page_config(page_title="MasterMedia AI Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .tool-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #eee;
        text-align: center;
        transition: 0.3s;
    }
    .tool-icon { font-size: 40px; }
    .tool-title { font-weight: bold; font-size: 20px; color: #333; margin: 10px 0; }
    </style>
    """, unsafe_allow_html=True)

# စာမျက်နှာ ထိန်းချုပ်ရန် session_state သုံးခြင်း
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- HOME PAGE (DASHBOARD) ---
if st.session_state.page == 'home':
    st.title("🚀 MasterMedia Workflow V4.5")
    st.write("Professional AI Toolset for Content Creators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="tool-card"><div class="tool-icon">🎬</div><div class="tool-title">Video Recap</div><p style="color:gray;">ဗီဒီယိုများကို အလိုအလျောက် အနှစ်ချုပ်ပေးမည်</p></div>', unsafe_allow_html=True)
        if st.button("Open Recap Tool", use_container_width=True):
            st.session_state.page = 'recap'
            st.rerun()

# --- RECAP TOOL PAGE ---
elif st.session_state.page == 'recap':
    if st.button("⬅️ Back to Dashboard"):
        st.session_state.page = 'home'
        st.rerun()
        
    st.title("🎬 Professional Video Recapper")
    
    @st.cache_resource
    def load_ai():
        return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    uploaded_file = st.file_uploader("ဗီဒီယို တင်ပေးပါ...", type=["mp4", "mov"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.video(uploaded_file)
        
        if st.button("Generate Recap Script"):
            with st.spinner('AI က ဗီဒီယိုကို လေ့လာနေပါသည်...'):
                try:
                    cap_model = load_ai()
                    vidcap = cv2.VideoCapture(tfile.name)
                    success, frame = vidcap.read()
                    if success:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        result = cap_model(pil_img)
                        st.subheader("AI Result:")
                        st.success(result[0]['generated_text'])
                    vidcap.release()
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.unlink(tfile.name)
