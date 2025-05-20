import streamlit as st
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import tempfile
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np
import wave
import os

# -------------------- Load the model --------------------
@st.cache_resource
def load_model():
    return EncoderClassifier.from_hparams(
        source="speechbrain/accent-id-ecapa",
        savedir="tmp_accent_id",
        use_auth_token=True  # Requires huggingface-cli login or token in env
    )

# Attempt to load the model
try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="Accent Classifier", page_icon="🎙️")
st.title("🎙️ Accent Classifier AI")
st.write("Upload an English speech audio file or record from your microphone to detect the accent.")

# -------------------- Helper Function --------------------
def classify_and_display(path):
    with st.spinner("Analyzing..."):
        try:
            if os.path.exists(path):
                prediction = model.classify_file(path)
                st.success(f"✅ Detected Accent: **{prediction[0]}**")
            else:
                st.error(f"❌ File not found at {path}")
        except Exception as e:
            st.error(f"❌ Error during classification: {e}")

# -------------------- File Upload Section --------------------
st.header("📁 Upload Audio File")
uploaded_file = st.file_uploader("Choose a .wav or .mp3 file", type=["wav", "mp3"])

if uploaded_file:
    file_extension = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(tmp_path, format=uploaded_file.type)
    classify_and_display(tmp_path)
    os.remove(tmp_path)

# -------------------- Microphone Input Section --------------------
st.header("🎤 Or Record Using Microphone")

class AudioProcessor:
    def __init__(self):
        self.frames = []

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        audio = frame.to_ndarray()
        self.frames.append(audio)
        return frame

processor = AudioProcessor()

ctx = webrtc_streamer(
    key="mic",
    audio_receiver_size=1024,
    media_stream_constraints={"audio": True, "video": False},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    audio_frame_callback=processor.recv
)

if not ctx.state.playing and processor.frames:
    try:
        audio_data = b''.join([frame.astype(np.int16).tobytes() for frame in processor.frames])
        wav_path = "mic_recording.wav"
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_data)

        st.audio(wav_path)
        classify_and_display(wav_path)
        os.remove(wav_path)

    except Exception as e:
        st.error(f"❌ Error processing microphone audio: {e}")
