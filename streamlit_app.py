import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Video Recap Tool", page_icon="🎬")
st.title("🎬 Video Recap Tool")

# AI Model ကို ခေါ်ယူခြင်း (Task နာမည်ကို image-text-to-text သို့ ပြောင်းထားသည်)
@st.cache_resource
def load_model():
    return pipeline("image-text-to-text", model="Salesforce/blip-image-captioning-base")

caption_model = load_model()

uploaded_file = st.file_uploader("ဗီဒီယို တင်ပါ", type=["mp4", "mov", "avi"])

if uploaded_file:
    # ဗီဒီယိုဖိုင်ကို ယာယီသိမ်းဆည်းရန်
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.video("temp_video.mp4")
    
    if st.button("Generate Recap"):
        with st.spinner('AI က ဗီဒီယိုကို ကြည့်နေပါသည်...'):
            try:
                # Video Frame တစ်ခုကို Recap လုပ်ခြင်း
                result = caption_model("temp_video.mp4")
                st.subheader("AI Recap Result:")
                st.success(result[0]['generated_text'])
                st.info("အပေါ်က အင်္ဂလိပ်စာကို Copy ကူးပြီး ကျွန်တော့်ဆီ ပို့ပေးပါ။ မြန်မာလို Recap ပြန်ရေးပေးပါ့မယ်။")
            except Exception as e:
                st.error(f"Error တက်သွားပါသည်: {e}")
