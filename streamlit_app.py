import streamlit as st
from transformers import pipeline
from PIL import Image
import cv2
import tempfile
import os

st.set_page_config(page_title="Video Recap Tool", page_icon="🎬")
st.title("🎬 Video Recap Tool")

# AI Model ကို လုံးဝ အခြေခံကျကျ ခေါ်ယူခြင်း
@st.cache_resource
def load_model():
    # task နာမည်ကို model တိုက်ရိုက်ခေါ်ခြင်းဖြင့် error ရှောင်ပါမည်
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

caption_model = load_model()

uploaded_file = st.file_uploader("ဗီဒီယို တင်ပါ", type=["mp4", "mov", "avi"])

if uploaded_file:
    # ဗီဒီယိုကို ယာယီသိမ်းဆည်းခြင်း
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    st.video(uploaded_file)
    
    if st.button("Generate Recap"):
        with st.spinner('AI က ဗီဒီယိုကို ကြည့်နေပါသည်...'):
            try:
                # ဗီဒီယိုထဲမှ ပုံရိပ်ကို OpenCV ဖြင့် ထုတ်ယူခြင်း
                vidcap = cv2.VideoCapture(tfile.name)
                # ဗီဒီယိုရဲ့ ပထမဆုံး စက္ကန့်ပိုင်းပုံကို ယူပါမည်
                success, image = vidcap.read()
                
                if success:
                    # BGR မှ RGB သို့ပြောင်းပြီး PIL format လုပ်ခြင်း
                    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    
                    # AI Recap ထုတ်ပေးခြင်း
                    result = caption_model(pil_img)
                    recap_text = result[0]['generated_text']
                    
                    st.subheader("AI Recap Result:")
                    st.success(recap_text)
                    st.info("💡 ဒီစာသားလေးကို ကျွန်တော့်ဆီ ပို့ပေးပါ။ မြန်မာလို Recap Script အပီအပြင် ရေးပေးပါ့မယ်။")
                else:
                    st.error("ဗီဒီယိုပုံရိပ်ကို ဖတ်၍မရပါ။")
            except Exception as e:
                st.error(f"Error ဖြစ်ပွားခဲ့သည်- {str(e)}")
            finally:
                vidcap.release()
                os.unlink(tfile.name)
