import streamlit as st
import requests
import cv2
import tempfile
import os

# Professional Dashboard UI Setup
st.set_page_config(page_title="Master YourMedia Workflow", layout="wide")

# CSS ဖြင့် UI ကို ပုံစံတူအောင် အလှဆင်ခြင်း
st.markdown("""
    <style>
    .main { background-color: #fcfcfc; }
    .stButton>button {
        width: 100%; border-radius: 12px; height: 100px;
        background-color: white; border: 1px solid #e0e0e0;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.02);
    }
    .stButton>button:hover { border-color: #6c5ce7; color: #6c5ce7; }
    .tool-header { text-align: center; margin-bottom: 30px; }
    </style>
    """, unsafe_allow_html=True)

if 'page' not in st.session_state:
    st.session_state.page = 'home'

# --- NAVIGATION LOGIC ---
def go_home(): st.session_state.page = 'home'

# --- HOME PAGE (DASHBOARD) ---
if st.session_state.page == 'home':
    st.markdown("<div class='tool-header'><h1>Master YourMedia Workflow.</h1><p>PROFESSIONAL AI TOOLSET V4.5</p></div>", unsafe_allow_html=True)
    
    # Tool များကို Column လိုက် ခွဲပြခြင်း
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎬 \n Video Recap \n (AI Analysis)"): st.session_state.page = 'recap'
        if st.button("📝 \n Content Creator \n (TikTok/Reels)"): st.session_state.page = 'creator'
        
    with col2:
        if st.button("🌍 \n Translate \n (Multi-Language)"): st.session_state.page = 'translate'
        if st.button("🎙️ \n Transcribe \n (Voice to Text)"): st.info("Coming Soon!")

# --- RECAP TOOL PAGE ---
elif st.session_state.page == 'recap':
    st.button("⬅️ Back to Dashboard", on_click=go_home)
    st.title("🎬 AI Video Recapper")
    
    video_file = st.file_uploader("ဗီဒီယို တင်ပေးပါ", type=["mp4", "mov"])
    
    if video_file:
        st.video(video_file)
        if st.button("Generate Recap Script"):
            with st.spinner('AI က ဗီဒီယိုကို စစ်ဆေးနေသည်...'):
                try:
                    # Video Frame ထုတ်ယူခြင်း
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(video_file.read())
                    cap = cv2.VideoCapture(tfile.name)
                    success, frame = cap.read()
                    
                    if success:
                        _, img_encoded = cv2.imencode('.jpg', frame)
                        # Hugging Face API (Free tier) သုံး၍ Recap လုပ်ခြင်း
                        API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
                        response = requests.post(API_URL, data=img_encoded.tobytes())
                        output = response.json()
                        
                        st.subheader("AI Recap Result:")
                        st.success(output[0]['generated_text'])
                        st.info("💡 အပေါ်ကစာကို Copy ကူးပြီး Content Creator ထဲမှာ Script ပြန်ရေးခိုင်းပါ။")
                    cap.release()
                    os.unlink(tfile.name)
                except:
                    st.error("API ခေတ္တ မအားသေးပါ။ နောက်တစ်ခေါက် ပြန်နှိပ်ပေးပါ။")

# --- CONTENT CREATOR PAGE ---
elif st.session_state.page == 'creator':
    st.button("⬅️ Back to Dashboard", on_click=go_home)
    st.title("📝 AI Content Creator")
    
    recap_input = st.text_area("Recap စာသား ထည့်ပါ (သို့မဟုတ်) ခေါင်းစဉ် ရိုက်ပါ")
    if st.button("Write TikTok Script"):
        st.subheader("သင့်အတွက် Script:")
        st.write(f"🎬 **TikTok Recap Script**\n\n[Hook]: အားလုံးပဲ မင်္ဂလာပါ! ဒီနေ့မှာတော့ {recap_input} အကြောင်း ပြောပြမယ်...\n\n[Body]: ဗီဒီယိုထဲမှာ တွေ့ရတဲ့အတိုင်းပဲ... (အသေးစိတ်ရှင်းပြချက်)\n\n[Call to Action]: နောက်ထပ် ဘာတွေ ကြည့်ချင်သေးလဲ? Comment မှာ ရေးခဲ့ဦးနော်!")
