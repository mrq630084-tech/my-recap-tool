import streamlit as st
from transformers import pipeline

st.title("🎬 Video Recap Tool")

# Task နာမည်ကို 'image-text-to-text' သို့မဟုတ် 'visual-question-answering' သို့ ပြောင်းပါမည်
@st.cache_resource
def load_model():
    # task နာမည်ကို model နှင့် ကိုက်ညီအောင် 'image-text-to-text' ဟု ပြောင်းလဲထားပါသည်
    return pipeline("image-text-to-text", model="Salesforce/blip-image-captioning-base")

caption_model = load_model()

uploaded_file = st.file_uploader("ဗီဒီယို တင်ပါ", type=["mp4", "mov", "avi"])

if uploaded_file:
    # ဗီဒီယိုကို ယာယီသိမ်းဆည်းခြင်း
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.video("temp_video.mp4")
    
    if st.button("Generate Recap"):
        with st.spinner('AI က ဗီဒီယိုကို ကြည့်နေပါသည်...'):
            try:
                # Video file ကို model ထဲ ထည့်သွင်းစဉ် Frame တစ်ခုကို recap လုပ်ပေးပါမည်
                result = caption_model("temp_video.mp4")
                st.success(f"Recap (English): {result[0]['generated_text']}")
                st.info("အပေါ်က စာသားကို Copy ကူးပြီး ကျွန်တော့်ဆီ ပို့ပေးပါ။ မြန်မာလို Recap Script ရေးပေးပါ့မယ်။")
            except Exception as e:
                st.error(f"Error တက်သွားပါသည်- {e}")
