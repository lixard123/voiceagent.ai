import streamlit as st
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import tempfile
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np
import wave
import os

# Load the model once
@st.cache_resource
def load_model():
    return EncoderClassifier.from_hparams(
        source="speechbrain/accent-id-ecapa",
        savedir="tmp_accent_id"
    )

model = load_model()

st.set_page_config(page_title="Accent Classifier", page_icon="🎙️")
st.title("🎙️ Accent Classifier AI")
st.write("Upload an English speech audio file or record from your microphone to detect the accent.")

# ---- Upload audio section ----
st.header("📁 Upload Audio File")
uploaded_file = st.file_uploader("Choose a .wav or .mp3 file", type=["wav", "mp3"])

def classify_and_display(path):
    with st.spinner("Analyzing..."):
        prediction = model.classify_file(path)
        st.success(f"✅ Detected Accent: **{prediction[3]}**")

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    classify_and_display(tmp_path)

# ---- Microphone input section ----
st.header("🎤 Or Record Using Microphone")

class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

ctx = webrtc_streamer(
    key="mic",
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    audio_frame_callback=AudioProcessor().recv
)

if ctx.state.playing:
    st.warning("Recording... speak now.")
elif ctx.audio_receiver:
    audio_data = b''.join([
        frame.to_ndarray().astype(np.int16).tobytes()
        for frame in ctx.audio_receiver.get_frames(timeout=1)
    ])
    wav_path = "mic_recording.wav"
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_data)

    st.audio(wav_path)
    classify_and_display(wav_path)
