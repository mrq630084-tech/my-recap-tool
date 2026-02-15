import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(page_title="Video Recap Tool", page_icon="🎬")
st.title("🎬 Video Recap Tool")

# AI Model ကို ပိုမိုမြန်ဆန်ပြီး Error ကင်းအောင် ခေါ်ယူခြင်း
@st.cache_resource
def load_model():
    # dtype ပြဿနာကို ရှောင်ရှားရန် device setting ထည့်ထားသည်
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=device)

caption_model = load_model()

uploaded_file = st.file_uploader("ဗီဒီယို တင်ပါ", type=["mp4", "mov", "avi"])

if uploaded_file:
    # ဗီဒီယိုဖိုင်ကို ယာယီသိမ်းခြင်း
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.video("temp_video.mp4")
    
    if st.button("Generate Recap"):
        with st.spinner('AI က ဗီဒီယိုကို ကြည့်နေပါသည်...'):
            try:
                # Video Analysis ပြုလုပ်ခြင်း
                # dtype error ကင်းစေရန် default parameter များဖြင့် run ပါသည်
                result = caption_model("temp_video.mp4")
                
                st.subheader("AI Recap Result (English):")
                recap_text = result[0]['generated_text']
                st.success(recap_text)
                
                st.divider()
                st.info("💡 ဒီအင်္ဂလိပ်စာသားကို Copy ကူးပြီး ကျွန်တော့်ဆီ ပို့ပေးပါ။ TikTok အတွက် အလန်းစား Recap Script မြန်မာလို ပြန်ရေးပေးပါ့မယ်။")
            except Exception as e:
                st.error(f"Error ဖြစ်ပွားခဲ့သည်- {str(e)}")
                st.warning("အကြံပြုချက်- ဗီဒီယိုဖိုင် အရမ်းမကြီးစေရန် သတိပြုပေးပါ။")
