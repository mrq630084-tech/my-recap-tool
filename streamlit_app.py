import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import tempfile
import os
import yt_dlp
from PIL import Image

# ၁။ UI Layout အလှဆင်ခြင်း
st.set_page_config(page_title="Master YourMedia Workflow", layout="wide")

st.markdown("""
    <style>
    .stTextInput>div>div>input { border-radius: 12px; padding: 12px; border: 2px solid #6c5ce7; }
    .stButton>button { width: 100%; border-radius: 12px; height: 55px; background-color: #6c5ce7; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# ၂။ AI Model Loading
@st.cache_resource
def load_ai():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

st.title("🚀 Master YourMedia Workflow.")
st.subheader("Link to AI Script Converter")

# Link ထည့်သည့်နေရာ
video_url = st.text_input("YouTube သို့မဟုတ် Video Link ကို ဒီမှာထည့်ပါ...", placeholder="https://youtube.com/shorts/...")

if st.button("Generate Script from Link"):
    if video_url:
        with st.spinner('AI က ဗီဒီယိုကို ချိတ်ဆက်နေပါသည်...'):
            try:
                # Bot စစ်ဆေးမှုကို ကျော်ဖြတ်ရန် yt-dlp option များ
                ydl_opts = {
                    'format': 'best[ext=mp4]/best',
                    'quiet': True,
                    'no_warnings': True,
                    'outtmpl': 'temp_vid.mp4'
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])
                
                processor, model = load_ai()
                vidcap = cv2.VideoCapture('temp_vid.mp4')
                success, frame = vidcap.read()
                
                if success:
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    inputs = processor(Image.fromarray(img_rgb), return_tensors="pt")
                    out = model.generate(**inputs)
                    recap_text = processor.decode(out[0], skip_special_tokens=True)
                    
                    # ၃။ Professional Script ထုတ်ပေးခြင်း
                    st.divider()
                    st.success("✅ Script Generated Successfully!")
                    
                    final_script = f"""
                    🎬 **TikTok/Reels Recap Script**
                    
                    [Intro]: ဟေး... အားလုံးပဲ မင်္ဂလာပါ! ဒီနေ့ ဗီဒီယိုလေးမှာ ဘာတွေပါမလဲဆိုတော့...
                    [Body]: ဗီဒီယိုရဲ့ အဓိက မြင်ကွင်းကတော့ {recap_text} ပဲ ဖြစ်ပါတယ်။ ဒါဟာ တကယ်ကို စိတ်ဝင်စားဖို့ကောင်းတဲ့ အခိုက်အတန့်ပါပဲ။
                    [Outro]: ဒီလိုမျိုး ဗီဒီယိုတွေ ထပ်ကြည့်ချင်ရင်တော့ Follow လုပ်ထားဖို့ မမေ့နဲ့ဦးနော်။
                    """
                    st.text_area("သင့်အတွက် Script အချောသတ် -", value=final_script, height=250)
                
                vidcap.release()
                if os.path.exists('temp_vid.mp4'): os.remove('temp_vid.mp4')
                
            except Exception as e:
                st.error("Error: YouTube ကန့်သတ်ချက်ကြောင့် Link ကို ဖတ်မရဖြစ်နေပါသည်။ Link အမှန်ဖြစ်ကြောင်း ထပ်မံစစ်ဆေးပေးပါ။")
    else:
        st.warning("ကျေးဇူးပြု၍ Link အရင်ထည့်ပေးပါဗျ။")
