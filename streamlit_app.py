import streamlit as st
from transformers import pipeline
from PIL import Image
import cv2
import tempfile
import os

st.title("🎬 Video Recap Tool")

# AI Model ကို လုံးဝ ရိုးရိုးရှင်းရှင်းပဲ ခေါ်ပါမယ်
@st.cache_resource
def load_model():
    # task နာမည်ကို 'image-to-text' ဟုပဲ ထားပါမည် (Error မတက်စေရန် version အဟောင်းအတိုင်း သုံးပါမည်)
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

caption_model = load_model()

uploaded_file = st.file_uploader("ဗီဒီယို တင်ပါ", type=["mp4", "mov", "avi"])

if uploaded_file:
    # ဗီဒီယိုကို ခဏ သိမ်းထားရန်
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    st.video(uploaded_file)
    
    if st.button("Generate Recap"):
        with st.spinner('AI က ဗီဒီယိုကို ကြည့်နေပါသည်...'):
            try:
                # ဗီဒီယိုထဲက ပုံကို ဆွဲထုတ်ခြင်း (OpenCV သုံးထားသည်)
                vidcap = cv2.VideoCapture(tfile.name)
                success, image = vidcap.read()
                
                if success:
                    # ပုံကို AI နားလည်အောင် ပြောင်းခြင်း
                    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    
                    # AI ကို Recap ထုတ်ခိုင်းခြင်း
                    result = caption_model(pil_img)
                    recap_text = result[0]['generated_text']
                    
                    st.subheader("AI Recap Result:")
                    st.success(recap_text)
                else:
                    st.error("ဗီဒီယိုထဲက ပုံရိပ်ကို ဖတ်လို့မရပါဘူး။")
            except Exception as e:
                st.error(f"Error တက်သွားပါသည်- {e}")
            finally:
                vidcap.release()
                os.unlink(tfile.name)
