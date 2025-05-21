# app.py
import streamlit as st
from transformers import pipeline
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
from streamlit_webrtc import webrtc_streamer, WebRtcMode 
import av
import numpy as np
import tempfile
import os
import torch
import torchaudio

st.set_page_config(page_title="English Accent Classifier", layout="centered")
st.title("🎙️ English Accent Classification")
st.markdown("Upload an audio file or record via mic to classify your English accent.")

# Load model once
@st.cache_resource
def load_model():
    return pipeline("audio-classification", model="dima806/english_accents_classification")

pipe = load_model()

# ---- Audio Processor for Mic ----
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

    def get_audio(self):
        if self.frames:
            audio = np.concatenate(self.frames, axis=1)
            self.frames = []
            return audio
        return None

# ---- UI for Input Mode ----
input_mode = st.radio("Select Input Mode", ["Upload Audio File", "Record from Microphone"])

audio_path = None

if input_mode == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload an audio file (.wav or .mp3)", type=["wav", "mp3"])
    if uploaded_file:
        st.audio(uploaded_file)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name

elif input_mode == "Record from Microphone":
    st.info("Click 'Start' and speak into the mic.")
    ctx = webrtc_streamer(
        key="mic",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        #client_settings={"media_stream_constraints": {"audio": True, "video": False}},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
        audio_processor_factory=AudioProcessor,
    )

    if ctx.state.playing:
        if st.button("Save Recording and Run Classification"):
            audio = ctx.audio_processor.get_audio()
            if audio is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    # Save as mono-channel, 16kHz
                    torchaudio.save(tmp.name, torch.tensor(audio).float(), 16000)
                    audio_path = tmp.name
            else:
                st.warning("No audio recorded yet.")

# ---- Run Inference ----
if st.button("Run Accent Classification") and audio_path:
    with st.spinner("Running model..."):
        try:
            results = pipe(audio_path)
            st.success("Prediction complete!")
            st.write("### 🎧 Top Predictions:")
            for r in results:
                st.write(f"**{r['label']}**: {r['score']:.2%}")
        except Exception as e:
            st.error(f"Error during inference: {e}")
        finally:
            os.remove(audio_path)
