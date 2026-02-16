import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
import torch

st.set_page_config(page_title="Video Recap Tool", layout="centered")

st.title("🎬 Video Recap Tool")
st.write("Upload an image → get caption → recap summary")

@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    summarizer = pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

    return processor, model, summarizer

processor, model, summarizer = load_models()

uploaded = st.file_uploader(
    "Upload image frame from video",
    type=["jpg", "jpeg", "png"]
)

if uploaded:

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Frame")

    with st.spinner("Generating caption..."):
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

    st.subheader("📌 Caption")
    st.write(caption)

    with st.spinner("Creating recap summary..."):
        prompt = f"Summarize this scene simply: {caption}"
        result = summarizer(
            prompt,
            max_length=60,
            do_sample=False
        )
        summary = result[0]["generated_text"]

    st.subheader("🎥 Recap")
    st.write(summary)

    st.success("Done!")
