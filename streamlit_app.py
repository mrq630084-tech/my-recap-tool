import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Video Recap Tool")
st.title("🎬 AI Video Recap Generator")

# ဗီဒီယိုမတင်ခင် model ကို ကြိုမခေါ်အောင် လုပ်ထားပါတယ်
@st.cache_resource
def load_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

uploaded_file = st.file_uploader("ဗီဒီယို တင်ပေးပါ...", type=["mp4", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    
    if st.button("Recap ထုတ်မည်"):
        with st.spinner('ခဏစောင့်ပေးပါ၊ AI က ကြည့်နေပါတယ်...'):
            try:
                # ဗီဒီယိုတင်ပြီးမှသာ model ကို စခေါ်မှာပါ
                model = load_model()
                # ယာယီအားဖြင့် ဖိုင်နာမည်ကိုသုံးပြီး စစ်ခိုင်းပါမယ်
                result = model(uploaded_file.name)
                
                st.subheader("AI Recap (English):")
                st.success(result[0]['generated_text'])
                st.info("အပေါ်က အင်္ဂလိပ်စာကို Copy ကူးပြီး ကျွန်တော့်ဆီ ပို့ပေးပါ။ မြန်မာလို Recap Script ပြန်ရေးပေးပါ့မယ်။")
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("ဗီဒီယိုဖိုင်တစ်ခု အရင်ရွေးချယ်ပေးပါခင်ဗျာ။")
