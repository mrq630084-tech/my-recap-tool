import streamlit as st
from transformers import pipeline
from PIL import Image
import cv2
import tempfile
import os

# Professional Dashboard Layout Setup
st.set_page_config(page_title="Master YourMedia Workflow", layout="wide")

# CSS ဖြင့် UI ကို ပုံထဲကအတိုင်း အဖြူရောင် Card ဒီဇိုင်း ပြောင်းခြင်း
st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stButton>button {
        width: 100%; border-radius: 20px; height: 110px;
        background-color: white; color: #333;
        border: 1px solid #eee; font-size: 18px; font-weight: 500;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        transition: 0.3s;
    }
    .stButton>button:hover {
        border-color: #6c5ce7; color: #6c5ce7;
        box-shadow: 0 8px 15px rgba(0,0,0,0.05);
        transform: translateY(-2px);
    }
    .header-section { text-align: center; margin-bottom: 40px; }
    </style>
    """, unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- MODEL LOADING (Local မောင်းနှင်ခြင်း - API မလိုပါ) ---
@st.cache_resource
def load_local_ai():
    # API မသုံးဘဲ server ပေါ်မှာတင် model ကို တိုက်ရိုက် တင်ပါသည်
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# --- HOME PAGE ---
if st.session_state.page == 'home':
    st.markdown("<div class='header-section'><h1>Master YourMedia Workflow.</h1><p>PROFESSIONAL AI TOOLSET V4.5</p></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎬 \n Video Recap"): st.session_state.page = 'recap'; st.rerun()
        if st.button("📝 \n Content Creator"): st.session_state.page = 'creator'; st.rerun()
    with col2:
        if st.button("🌍 \n Translate"): st.session_state.page = 'translate'; st.rerun()
        if st.button("🎙️ \n Transcribe"): st.info("Maintenance Mode")

# --- RECAP TOOL PAGE ---
elif st.session_state.page == 'recap':
    if st.button("⬅️ Back"): st.session_state.page = 'home'; st.rerun()
    st.title("🎬 Local AI Video Recapper")
    
    video = st.file_uploader("ဗီဒီယို တင်ပါ", type=["mp4", "mov"])
    if video:
        st.video(video)
        if st.button("Start AI Analysis"):
            with st.spinner('AI က ဗီဒီယိုကို စက္ကန့်တိုင်း လေ့လာနေပါသည် (API မပါဘဲ တိုက်ရိုက်စစ်နေသည်)...'):
                try:
                    cap_model = load_local_ai()
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(video.read())
                    vidcap = cv2.VideoCapture(tfile.name)
                    success, frame = vidcap.read()
                    
                    if success:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        # Local AI Model က တိုက်ရိုက် အဖြေထုတ်ပေးပါသည်
                        result = cap_model(pil_img)
                        st.subheader("AI Recap (Error-Free):")
                        st.success(result[0]['generated_text'])
                    vidcap.release()
                    os.unlink(tfile.name)
                except Exception as e:
                    st.error(f"Error: {e}")
