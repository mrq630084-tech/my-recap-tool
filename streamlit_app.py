import streamlit as st
import requests
import cv2
import tempfile
import os

st.set_page_config(page_title="Video Recap Tool", page_icon="🎬")
st.title("🎬 Video Recap Tool")

# Hugging Face API သုံးပြီး Recap လုပ်မည့် Function
def get_recap(image_data):
    # BLIP model ကို API မှတစ်ဆင့် လှမ်းသုံးခြင်း
    API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    # သင်၏ Hugging Face Token ရှိလျှင် အောက်ပါနေရာတွင် ထည့်နိုင်သည် (မရှိလည်း အကြိမ်ရေအနည်းငယ် စမ်းသပ်နိုင်သည်)
    headers = {"Authorization": "Bearer hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"} 
    
    response = requests.post(API_URL, data=image_data)
    return response.json()

uploaded_file = st.file_uploader("ဗီဒီယို တင်ပါ", type=["mp4", "mov", "avi"])

if uploaded_file:
    # ဗီဒီယိုကို ယာယီသိမ်းခြင်း
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    st.video(uploaded_file)
    
    if st.button("Generate Recap"):
        with st.spinner('AI က ဗီဒီယိုကို လေ့လာနေပါသည်...'):
            try:
                # ဗီဒီယိုထဲမှ ပုံရိပ်ကို ထုတ်ယူခြင်း
                vidcap = cv2.VideoCapture(tfile.name)
                success, frame = vidcap.read()
                if success:
                    # ပုံကို byte အဖြစ်ပြောင်းခြင်း
                    _, img_encoded = cv2.imencode('.jpg', frame)
                    img_bytes = img_encoded.tobytes()
                    
                    # API ကို ပို့ခြင်း
                    output = get_recap(img_bytes)
                    
                    # ရလဒ် ထုတ်ပြခြင်း
                    if isinstance(output, list) and len(output) > 0:
                        recap_text = output[0].get('generated_text', 'စာသားရှာမတွေ့ပါ')
                        st.subheader("AI Recap (English):")
                        st.success(recap_text)
                        st.info("💡 ဒီအင်္ဂလိပ်စာကို Copy ကူးပြီး ကျွန်တော့်ဆီ ပို့ပေးပါ။ မြန်မာလို Recap Script ပြန်ရေးပေးပါ့မယ်။")
                    else:
                        st.error("AI Model အလုပ်လုပ်ပုံ အနည်းငယ် ကြန့်ကြာနေပါသည်။ ခဏနေပြန်စမ်းပေးပါ။")
                else:
                    st.error("ဗီဒီယိုကို ဖတ်၍မရပါ။")
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                vidcap.release()
                os.unlink(tfile.name)
