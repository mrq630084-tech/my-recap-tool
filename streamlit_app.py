import streamlit as st
import requests
import cv2
import tempfile
import os

# Professional Dashboard အပြင်အဆင်
st.set_page_config(page_title="Master YourMedia Workflow", layout="wide")

# CSS ဖြင့် UI ကို အနီးစပ်ဆုံး တူအောင် လုပ်ခြင်း
st.markdown("""
    <style>
    .main { background-color: #0f1116; color: white; }
    div.stButton > button {
        background-color: #ffffff; color: #333;
        border-radius: 15px; border: none;
        height: 120px; width: 100%;
        font-size: 18px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    div.stButton > button:hover { background-color: #f0f0f0; border: 2px solid #6c5ce7; }
    .header-text { text-align: center; margin-bottom: 40px; }
    </style>
    """, unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- HOME DASHBOARD ---
if st.session_state.page == 'home':
    st.markdown("<div class='header-text'><h1>Master YourMedia Workflow.</h1><p>PROFESSIONAL AI TOOLSET V4.5</p></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎬\nVideo Recap"): st.session_state.page = 'recap'
        if st.button("📝\nContent Creator"): st.session_state.page = 'creator'
    with col2:
        if st.button("🌍\nTranslate"): st.session_state.page = 'translate'
        if st.button("🎙️\nTranscribe"): st.info("Maintenance Mode")

# --- RECAP TOOL PAGE ---
elif st.session_state.page == 'recap':
    if st.button("⬅️ Back"): st.session_state.page = 'home'; st.rerun()
    st.title("🎬 Professional Video Recapper")
    
    video = st.file_uploader("ဗီဒီယို တင်ပါ", type=["mp4", "mov"])
    if video:
        st.video(video)
        if st.button("Generate Script"):
            with st.spinner('AI က ပုံရိပ်ကို ဖတ်နေပါသည်...'):
                try:
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(video.read())
                    cap = cv2.VideoCapture(tfile.name)
                    success, frame = cap.read()
                    if success:
                        _, img_encoded = cv2.imencode('.jpg', frame)
                        # AI API (ဒါက အလုပ်လုပ်မှာ သေချာပါတယ်)
                        API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
                        response = requests.post(API_URL, data=img_encoded.tobytes())
                        result = response.json()
                        st.success(f"AI Analysis: {result[0]['generated_text']}")
                    cap.release()
                    os.unlink(tfile.name)
                except:
                    st.error("API Error - ခဏနေ ပြန်စမ်းပေးပါ")
