import streamlit as st
from transformers import pipeline
from PIL import Image
import cv2
import tempfile
import os

# Dashboard UI Setup
st.set_page_config(page_title="Master YourMedia Workflow", layout="wide")

# CSS ဖြင့် UI ကို Card ပုံစံ ပြင်ခြင်း
st.markdown("""
    <style>
    .stButton>button {
        width: 100%; border-radius: 20px; height: 100px;
        background-color: white; color: #333; border: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- MODEL LOADING (Correcting the Task Name) ---
@st.cache_resource
def load_ai():
    # Error တက်ခဲ့တဲ့ image-to-text အစား image-text-to-text ကို သုံးပါမယ်
    return pipeline("image-text-to-text", model="Salesforce/blip-image-captioning-base")

# --- HOME PAGE ---
if st.session_state.page == 'home':
    st.title("🚀 Master YourMedia Workflow.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎬 \n Video Recap"): st.session_state.page = 'recap'; st.rerun()
    with col2:
        if st.button("📝 \n Content Creator"): st.session_state.page = 'creator'; st.rerun()

# --- RECAP TOOL PAGE ---
elif st.session_state.page == 'recap':
    if st.button("⬅️ Back"): st.session_state.page = 'home'; st.rerun()
    st.title("🎬 AI Video Recapper")
    
    video = st.file_uploader("ဗီဒီယို တင်ပါ", type=["mp4", "mov"])
    if video:
        st.video(video)
        if st.button("Start Analysis"):
            with st.spinner('AI က ဗီဒီယိုကို လေ့လာနေပါသည်...'):
                try:
                    # Model ခေါ်ယူခြင်း
                    model = load_ai()
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(video.read())
                    vidcap = cv2.VideoCapture(tfile.name)
                    success, frame = vidcap.read()
                    
                    if success:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        # AI အဖြေထုတ်ခြင်း
                        result = model(pil_img)
                        st.subheader("AI Recap Result:")
                        st.success(result[0]['generated_text'])
                    vidcap.release()
                    os.unlink(tfile.name)
                except Exception as e:
                    st.error(f"Error: {e}")
