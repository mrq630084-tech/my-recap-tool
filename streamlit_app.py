import streamlit as st
from transformers import pipeline
from PIL import Image
import cv2
import tempfile
import os

st.set_page_config(page_title="Video Recap Tool", page_icon="🎬")
st.title("🎬 Video Recap Tool")

# AI Model ကို ပိုပြီး Stable ဖြစ်အောင် ခေါ်ယူခြင်း
@st.cache_resource
def load_model():
    # ဤနေရာတွင် task နာမည်ကို version အားလုံးနှင့် ကိုက်အောင် ထားထားသည်
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

caption_model = load_model()

uploaded_file = st.file_uploader("ဗီဒီယို တင်ပါ", type=["mp4", "mov", "avi"])

if uploaded_file:
    # ဗီဒီယိုကို ယာယီသိမ်းဆည်းခြင်း
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    st.video(uploaded_file)
    
    if st.button("Generate Recap"):
        with st.spinner('AI က ဗီဒီယိုကို လေ့လာနေပါသည်...'):
            try:
                # ဗီဒီယိုထဲမှ အလယ် frame တစ်ခုကို ထုတ်ယူခြင်း (Error ကင်းစေရန်)
                vidcap = cv2.VideoCapture(tfile.name)
                # ဗီဒီယိုရဲ့ အလယ်ပုံကို ယူရန်
                fps = vidcap.get(cv2.CAP_PROP_FPS)
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 1)) # ၁ စက္ကန့်မြောက်ပုံကို ယူမည်
                
                success, image = vidcap.read()
                if success:
                    # OpenCV image ကို PIL image သို့ ပြောင်းခြင်း
                    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)
                    
                    # AI Recap ထုတ်ပေးခြင်း
                    result = caption_model(pil_img)
                    recap_text = result[0]['generated_text']
                    
                    st.subheader("AI Recap (English):")
                    st.success(recap_text)
                    st.info("💡 ဒီစာသားကို Copy ကူးပြီး ကျွန်တော့်ဆီ ပို့ပေးပါ။ Recap Script မြန်မာလို ရေးပေးပါ့မယ်။")
                else:
                    st.error("ဗီဒီယိုပုံရိပ်ကို ဖတ်၍မရပါ။")
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                vidcap.release()
                os.unlink(tfile.name)
