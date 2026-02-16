import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import cv2
import tempfile
import os

# ၁။ UI Layout နှင့် CSS
st.set_page_config(page_title="Master YourMedia Workflow", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div.stButton > button {
        background-color: white; border-radius: 12px; height: 90px; width: 100%;
        border: 1px solid #ddd; font-weight: bold; transition: 0.3s;
    }
    div.stButton > button:hover { border-color: #6c5ce7; color: #6c5ce7; }
    .social-link {
        display: inline-block; padding: 10px 20px; border-radius: 8px;
        color: white; text-decoration: none; font-weight: bold; margin: 5px;
    }
    .yt { background-color: #ff0000; }
    .tl { background-color: #0088cc; }
    </style>
    """, unsafe_allow_html=True)

if 'page' not in st.session_state: st.session_state.page = 'home'

# ၂။ AI Model Loading
@st.cache_resource
def load_ai():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# --- HOME PAGE ---
if st.session_state.page == 'home':
    st.markdown("<h1 style='text-align:center;'>Master YourMedia Workflow.</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎬\nVideo Recap"): st.session_state.page = 'recap'; st.rerun()
    with col2:
        if st.button("📝\nContent Creator"): st.session_state.page = 'creator'; st.rerun()

    st.divider()
    # Social Media Links Section
    st.markdown("<h3 style='text-align:center;'>Connect with Us</h3>", unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown('<a href="https://youtube.com" class="social-link yt">📺 YouTube Tutorial</a>', unsafe_allow_html=True)
    with sc2:
        st.markdown('<a href="https://t.me" class="social-link tl">✈️ Telegram Channel</a>', unsafe_allow_html=True)

# --- RECAP & SCRIPT PAGE ---
elif st.session_state.page == 'recap':
    if st.button("⬅️ Back"): st.session_state.page = 'home'; st.rerun()
    st.title("🎬 Video to Script Converter")
    
    video = st.file_uploader("ဗီဒီယို တင်ပေးပါ", type=["mp4", "mov"])
    if video:
        st.video(video)
        if st.button("Generate Final Script"):
            with st.spinner('AI က ဇာတ်လမ်းကို ဖန်တီးနေပါသည်...'):
                try:
                    processor, model = load_ai()
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(video.read())
                    vidcap = cv2.VideoCapture(tfile.name)
                    
                    # ၅ စက္ကန့်ခြား တစ်ခါ ပုံဖတ်ခြင်း
                    success, frame = vidcap.read()
                    if success:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        inputs = processor(Image.fromarray(img_rgb), return_tensors="pt")
                        out = model.generate(**inputs)
                        raw_text = processor.decode(out[0], skip_special_tokens=True)
                        
                        # Professional Script အဖြစ် ပြောင်းလဲခြင်း
                        st.subheader("📝 TikTok/Reels Script:")
                        final_script = f"""
                        🎬 **Title: Life Moments Recap**
                        
                        [Hook]: သင်မမြင်ဖူးသေးတဲ့ မြင်ကွင်းတစ်ခုကို ပြပေးမယ်...
                        [Body]: ဒီဗီဒီယိုထဲမှာတော့ {raw_text} ကို တွေ့ရမှာဖြစ်ပြီး ဘဝရဲ့ အမှတ်တရတွေကို ဖော်ပြနေပါတယ်။
                        [Ending]: နောက်ထပ် ဗီဒီယိုတွေ ကြည့်ချင်ရင်တော့ Follow လုပ်ထားဖို့ မမေ့နဲ့ဦးနော်!
                        """
                        st.success(final_script)
                    vidcap.release()
                    os.unlink(tfile.name)
                except Exception as e:
                    st.error(f"Error: {e}")
