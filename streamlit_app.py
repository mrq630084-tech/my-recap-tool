import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import tempfile
import os
import yt_dlp

st.set_page_config(page_title="Master YourMedia Workflow", layout="wide")

# UI Design
st.markdown("""
    <style>
    .stTextInput>div>div>input { border-radius: 10px; padding: 15px; }
    .stButton>button { border-radius: 12px; height: 60px; background-color: #6c5ce7; color: white; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_ai():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

if 'page' not in st.session_state: st.session_state.page = 'home'

# --- HOME DASHBOARD ---
if st.session_state.page == 'home':
    st.title("🚀 Master YourMedia Workflow.")
    st.subheader("Link to AI Script Converter")
    
    # YouTube သို့မဟုတ် Video Link ထည့်ရန်နေရာ
    video_url = st.text_input("YouTube သို့မဟုတ် Video Link ကို ဒီမှာထည့်ပါ...", placeholder="https://www.youtube.com/watch?v=...")
    
    if st.button("Generate Script from Link"):
        if video_url:
            with st.spinner('AI က Link ထဲက ဗီဒီယိုကို လှမ်းဖတ်နေပါသည်...'):
                try:
                    # ၁။ Link မှ ဗီဒီယိုကို ခေတ္တဒေါင်းယူခြင်း
                    ydl_opts = {'format': 'best[ext=mp4]/best', 'outtmpl': 'temp_video.mp4'}
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([video_url])
                    
                    # ၂။ AI ဖြင့် Analysis လုပ်ခြင်း
                    processor, model = load_ai()
                    vidcap = cv2.VideoCapture('temp_video.mp4')
                    success, frame = vidcap.read()
                    
                    if success:
                        from PIL import Image
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        inputs = processor(Image.fromarray(img_rgb), return_tensors="pt")
                        out = model.generate(**inputs)
                        raw_text = processor.decode(out[0], skip_special_tokens=True)
                        
                        # ၃။ Professional Script ထုတ်ပေးခြင်း
                        st.divider()
                        st.subheader("📝 Generated Script:")
                        final_script = f"""
                        🎬 **Video Recap Script**
                        
                        [Hook]: ဒီဗီဒီယိုထဲမှာ ဘာတွေဖြစ်နေလဲဆိုတာ ကြည့်လိုက်ရအောင်...
                        [Content]: ဗီဒီယိုရဲ့ အဓိကအချက်ကတော့ {raw_text} ဖြစ်ပါတယ်။
                        [Call to Action]: အသေးစိတ်သိချင်ရင်တော့ Link ကို နှိပ်ပြီး ကြည့်နိုင်ပါတယ်!
                        """
                        st.success(final_script)
                        
                    vidcap.release()
                    if os.path.exists('temp_video.mp4'): os.remove('temp_video.mp4')
                except Exception as e:
                    st.error(f"Error: {e}. Link မှန်မမှန် ပြန်စစ်ပေးပါ။")
        else:
            st.warning("ကျေးဇူးပြု၍ Link တစ်ခု အရင်ထည့်ပေးပါဗျ။")
