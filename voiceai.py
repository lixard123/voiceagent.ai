import streamlit as st
from transformers import pipeline
import tempfile
import os
import torch
import torchaudio
import imageio_ffmpeg
import matplotlib.pyplot as plt

# Set environment variable for imageio-ffmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()
# Set torchaudio backend
torchaudio.set_audio_backend("soundfile")

# Initialize session state variables BEFORE anything else
if "audio_path" not in st.session_state:
    st.session_state.audio_path = None

# Set Streamlit page config
st.set_page_config(page_title="English Accent Classifier", layout="centered")
st.title("🎙️ English Accent Classification")
st.markdown("Upload an audio file to classify your English accent.")

@st.cache_resource
def load_model():
    with st.spinner("Loading accent classification model..."):
        return pipeline("audio-classification", model="dima806/english_accents_classification")

pipe = load_model()

uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])
if uploaded_file:
    st.audio(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        st.session_state.audio_path = tmp.name
    st.success("✅ Audio file uploaded and ready for classification.")

if st.session_state.audio_path:
    if st.button("Run Classification"):
        with st.spinner("🧠 Running model..."):
            try:
                results = pipe(st.session_state.audio_path)
                st.success("✅ Prediction complete!")
                st.write("### 🎧 Top Accent Predictions:")
                for r in results:
                    st.write(f"**{r['label']}**: {r['score']:.2%}")
            except Exception as e:
                st.error(f"❌ Error during inference: {e}")
            finally:
                try:
                    os.remove(st.session_state.audio_path)
                    st.session_state.audio_path = None
                except Exception as e:
                    print(f"Error cleaning up temp file: {e}")
