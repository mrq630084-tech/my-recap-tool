import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import cv2
import tempfile
import os

st.set_page_config(page_title="Master YourMedia Workflow", layout="wide")

# CSS UI Setup
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    div.stButton > button { border-radius: 12px; height: 80px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

if 'page' not in st.session_state: st.session_state.page = 'home'

if st.session_state.page == 'home':
    st.title("🚀 Master YourMedia Workflow.")
    if st.button("🎬 Video Recap"): st.session_state.page = 'recap'; st.rerun()

elif st.session_state.page == 'recap':
    if st.button("⬅️ Back"): st.session_state.page = 'home'; st.rerun()
    
    video = st.file_uploader("ဗီဒီယို တင်ပါ", type=["mp4", "mov"])
    if video:
        st.video(video)
        if st.button("Generate Detailed Recap"):
            with st.spinner('AI က ဗီဒီယိုတစ်ခုလုံးကို အသေးစိတ် လေ့လာနေပါသည်...'):
                try:
                    processor, model = load_blip_model()
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(video.read())
                    
                    vidcap = cv2.VideoCapture(tfile.name)
                    fps = vidcap.get(cv2.CAP_PROP_FPS) # ဗီဒီယိုရဲ့ Speed ကို ယူခြင်း
                    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    full_recap = []
                    # ၅ စက္ကန့်ခြား တစ်ခါ ပုံထုတ်မည် (5 * fps)
                    interval = int(fps * 5) 

                    for i in range(0, total_frames, interval):
                        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
                        success, frame = vidcap.read()
                        if success:
                            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_img = Image.fromarray(img_rgb)
                            inputs = processor(pil_img, return_tensors="pt")
                            out = model.generate(**inputs)
                            caption = processor.decode(out[0], skip_special_tokens=True)
                            timestamp = i // fps
                            full_recap.append(f"⏱️ {int(timestamp)}s: {caption}")
                    
                    st.subheader("အသေးစိတ် Recap ရလဒ်:")
                    for line in full_recap:
                        st.write(line) # စာကြောင်းအလိုက် ထုတ်ပေးခြင်း
                        
                    vidcap.release()
                    os.unlink(tfile.name)
                except Exception as e:
                    st.error(f"Error: {e}")
