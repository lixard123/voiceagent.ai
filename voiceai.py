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
    # Explicitly set use_auth_token=True to attempt authentication if needed
    # This helps if the model is gated or private.
    # Ensure you have a Hugging Face token set up in your environment
    # or have logged in via huggingface-cli login
    return EncoderClassifier.from_hparams(
        source="speechbrain/accent-id-ecapa",
        savedir="tmp_accent_id",
        use_auth_token=True # Add this argument
    )

# Catch potential errors during model loading
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop() # Stop the app if the model fails to load


st.set_page_config(page_title="Accent Classifier", page_icon="🎙️")
st.title("🎙️ Accent Classifier AI")
st.write("Upload an English speech audio file or record from your microphone to detect the accent.")

# ---- Upload audio section ----
st.header("📁 Upload Audio File")
uploaded_file = st.file_uploader("Choose a .wav or .mp3 file", type=["wav", "mp3"])

def classify_and_display(path):
    with st.spinner("Analyzing..."):
        try:
            # Ensure the file exists before trying to classify
            if os.path.exists(path):
                prediction = model.classify_file(path)
                st.success(f"✅ Detected Accent: **{prediction[3]}**")
            else:
                st.error(f"Error: File not found at {path}")
        except Exception as e:
            st.error(f"Error during classification: {e}")


if uploaded_file:
    # Streamlit's audio player might not handle all formats correctly for the underlying analysis.
    # It's better to save the file and then use the saved path for both audio display and classification.
    # Ensure the temporary file is saved with the correct extension based on the uploaded file type.
    file_extension = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(tmp_path, format=uploaded_file.type) # Use tmp_path for audio display
    classify_and_display(tmp_path)

    # Clean up the temporary file after processing
    os.remove(tmp_path)


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

# Only process microphone audio if the stream is inactive and frames were captured
if not ctx.state.playing and ctx.audio_receiver:
    try:
        # Retrieve frames from the audio receiver
        audio_frames = ctx.audio_receiver.get_frames(timeout=1)
        if audio_frames:
            audio_data = b''.join([
                frame.to_ndarray().astype(np.int16).tobytes()
                for frame in audio_frames
            ])
            wav_path = "mic_recording.wav"
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2) # 2 bytes per sample for int16
                wf.setframerate(16000) # Assuming a standard sample rate
                wf.writeframes(audio_data)

            st.audio(wav_path)
            classify_and_display(wav_path)

            # Clean up the microphone recording file
            os.remove(wav_path)
        else:
            st.warning("No audio frames captured from microphone.")
    except Exception as e:
        st.error(f"Error processing microphone audio: {e}")
