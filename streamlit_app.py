import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import tempfile
import os
import yt_dlp
from PIL import Image

# ၁။ UI Layout နှင့် CSS ပြင်ဆင်ခြင်း
st.set_page_config(page_title="Master YourMedia Workflow", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stTextInput>div>div>input { border-radius: 12px; padding: 12px; border: 2px solid #6c5ce7; }
    /* Card ပုံစံ ခလုတ်များ */
    div.stButton > button {
        background-color: white; color: #333; border-radius: 12px;
        height: 60px; width: 100%; border: 1px solid #ddd;
        font-weight: bold; box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    div.stButton > button:hover { border-color: #6c5ce7; color: #6c5ce7; }
    .header-section { text-align: center; margin-bottom: 30px; }
    </style>
    """, unsafe_allow_html=True)

# ၂။ AI Model Loading
@st.cache_resource
def load_ai_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- HOME DASHBOARD ---
if st.session_state.page == 'home':
    st.markdown("<div class='header-section'><h1>Master YourMedia Workflow.</h1><p>PROFESSIONAL AI TOOLSET V4.5</p></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎬 Video Recap"): st.session_state.page = 'recap'; st.rerun()
        if st.button("📝 Content Creator"): st.session_state.page = 'creator'; st.rerun()
    with col2:
        # YouTube နှင့် Telegram Link များ ထည့်သွင်းခြင်း
        st.link_button("📺 YouTube Tutorial", "https://youtube.com")
        st.link_button("✈️ Join Telegram", "https://t.me")

# --- RECAP FROM LINK PAGE ---
elif st.session_state.page == 'recap':
    if st.button("⬅️ Back to Dashboard"): st.session_state.page = 'home'; st.rerun()
    
    st.title("🎬 Link to AI Script Converter")
    # Link ထည့်သည့်နေရာ
    video_url = st.text_input("YouTube/Video Link ကို ဒီမှာထည့်ပါ...", placeholder="https://youtube.com/shorts/...")

    if st.button("Generate Script from Link"):
        if video_url:
            with st.spinner('AI က ဗီဒီယိုကို လေ့လာနေပါသည်...'):
                try:
                    # Bot စစ်ဆေးမှုကို ရှောင်ရှားရန် yt-dlp option များ
                    ydl_opts = {
                        'format': 'best[ext=mp4]/best',
                        'quiet': True,
                        'no_warnings': True,
                        'outtmpl': 'temp_vid.mp4',
                        'nocheckcertificate': True
                    }
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([video_url])
                    
                    processor, model = load_ai_model()
                    vidcap = cv2.VideoCapture('temp_vid.mp4')
                    success, frame = vidcap.read()
                    
                    if success:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        inputs = processor(Image.fromarray(img_rgb), return_tensors="pt")
                        out = model.generate(**inputs)
                        recap_text = processor.decode(out[0], skip_special_tokens=True)
                        
                        # Professional Script ထုတ်ပေးခြင်း
                        st.divider()
                        st.success("✅ Script ထွက်လာပါပြီ!")
                        
                        final_script = f"""
                        🎬 **Video Recap Script**
                        
                        [Intro/Hook]: ဟေး အားလုံးပဲ မင်္ဂလာပါ! ဒီနေ့ ဗီဒီယိုလေးမှာ ဘာတွေပါမလဲဆိုတော့...
                        [Main Content]: ဗီဒီယိုရဲ့ အဓိကမြင်ကွင်းကတော့ {recap_text} ပဲ ဖြစ်ပါတယ်။
                        [Ending]: နောက်ထပ် ဗီဒီယိုတွေ ထပ်ကြည့်ချင်ရင်တော့ Follow လုပ်ထားဖို့ မမေ့နဲ့ဦးနော်!
                        """
                        st.text_area("သင့်အတွက် Script အချောသတ် -", value=final_script, height=250)
                    
                    vidcap.release()
                    if os.path.exists('temp_vid.mp4'): os.remove('temp_vid.mp4')
                except Exception as e:
                    # Error message ပြသခြင်း
                    st.error(f"Error: Link ကို ဖတ်မရဖြစ်နေပါသည်။ YouTube ကန့်သတ်ချက်ကြောင့် ဖြစ်နိုင်ပါသည်။ Link ကို ပြန်စစ်ပေးပါ။")
        else:
            st.warning("Link တစ်ခု အရင်ထည့်ပေးပါဗျ။")
