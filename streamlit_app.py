import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import cv2
import tempfile
import os

# Professional Dashboard UI
st.set_page_config(page_title="Master YourMedia Workflow", layout="wide")

st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 15px; height: 100px; background-color: white; border: 1px solid #eee; }
    </style>
    """, unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- MODEL LOADING (The Most Stable Way) ---
@st.cache_resource
def load_blip_model():
    # Pipeline မသုံးဘဲ Model ကို တိုက်ရိုက်ခေါ်ခြင်းဖြင့် Error ရှောင်ပါမည်
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# --- HOME PAGE ---
if st.session_state.page == 'home':
    st.title("🚀 Master YourMedia Workflow.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎬 \n Video Recap"): st.session_state.page = 'recap'; st.rerun()
    with col2:
        if st.button("📝 \n Content Creator"): st.info("Recap ရမှ ဒါကို သုံးလို့ရပါမယ်")

# --- RECAP TOOL PAGE ---
elif st.session_state.page == 'recap':
    if st.button("⬅️ Back"): st.session_state.page = 'home'; st.rerun()
    st.title("🎬 AI Video Recapper (Fixed)")
    
    video = st.file_uploader("ဗီဒီယို တင်ပါ", type=["mp4", "mov"])
    if video:
        st.video(video)
        if st.button("Generate Script"):
            with st.spinner('AI က ဗီဒီယိုကို လေ့လာနေပါသည်...'):
                try:
                    processor, model = load_blip_model()
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(video.read())
                    
                    vidcap = cv2.VideoCapture(tfile.name)
                    success, frame = vidcap.read()
                    
                    if success:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        
                        # AI Processing (No Text Required here)
                        inputs = processor(pil_img, return_tensors="pt")
                        out = model.generate(**inputs)
                        caption = processor.decode(out[0], skip_special_tokens=True)
                        
                        st.subheader("AI Analysis Result:")
                        st.success(caption)
                    vidcap.release()
                    os.unlink(tfile.name)
                except Exception as e:
                    st.error(f"Error: {e}")
