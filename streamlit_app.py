import streamlit as st
from transformers import pipeline

st.title("🎬 Video Recap Tool")

@st.cache_resource
def load_model():
    return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

caption_model = load_model()

uploaded_file = st.file_uploader("ဗီဒီယို တင်ပါ", type=["mp4", "mov"])

if uploaded_file:
    st.video(uploaded_file)
    if st.button("Generate Recap"):
        with st.spinner('AI က ကြည့်နေပါသည်...'):
            result = caption_model(uploaded_file.name)
            st.success(f"Recap: {result[0]['generated_text']}")
