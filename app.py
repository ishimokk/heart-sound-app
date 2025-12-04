import streamlit as st
import numpy as np
import io

try:
    import librosa
except ImportError:
    librosa = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

st.set_page_config(page_title="Heart Sound Screening App", page_icon="🫀")

SAMPLE_RATE = 16000
CLIP_SECONDS = 5


@st.cache_resource(show_spinner=False)
def load_model(path: str = "heart_cnn_model.h5"):
    if tf is None:
        return None
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        return None


def load_and_fix_audio(file) -> np.ndarray:
    target_len = SAMPLE_RATE * CLIP_SECONDS

    if librosa is None:
        y = np.random.randn(target_len).astype("float32") * 0.01
        return y

    y, sr = librosa.load(file, sr=SAMPLE_RATE, mono=True)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    return y


def make_spectrogram(y: np.ndarray) -> np.ndarray:
    if librosa is None:
        spec = np.abs(np.fft.rfft(y))
        spec = spec[:512]
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        spec = spec.reshape(1, -1, 1, 1)
        return spec.astype("float32")

    S = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        window="hann",
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_norm[np.newaxis, ..., np.newaxis].astype("float32")


def model_predict(model, spec: np.ndarray):
    if model is not None:
        probs = model.predict(spec, verbose=0)[0]
        p_normal = float(probs[0])
        p_abnormal = float(probs[1]) if len(probs) > 1 else 1.0 - p_normal
    else:
        energy = float(spec.mean())
        p_abnormal = min(max((energy - 0.4) * 2.0, 0.0), 1.0)
        p_normal = 1.0 - p_abnormal

    label = "Likely NORMAL" if p_normal >= p_abnormal else "Possible ABNORMAL"
    return label, p_normal, p_abnormal


def main():
    st.title("🫀 Heart Sound Screening App")
    st.write(
        "This demo app screens recorded heart sounds and classifies them as "
        "`normal` or `possible abnormal` using a convolutional neural network (CNN). "
        "It is **not** a medical diagnosis tool."
    )

    audio_file = st.file_uploader(
        "Upload a heart sound recording (.wav recommended)",
        type=["wav", "mp3", "ogg"],
    )

    model = load_model()

    if model is None:
        st.info(
            "Running in **demo mode**. To use your real trained model, upload "
            "`heart_cnn_model.h5` into this repo and make sure `tensorflow` is "
            "listed in requirements.txt."
        )

    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")

        if st.button("Analyze heart sound"):
            with st.spinner("Analyzing..."):
                y = load_and_fix_audio(audio_file)
                spec = make_spectrogram(y)
                label, p_norm, p_abn = model_predict(model, spec)

            st.subheader("Result")
            st.markdown(f"### **{label}**")
            col1, col2 = st.columns(2)
            col1.metric("Probability normal", f"{p_norm:.2f}")
            col2.metric("Probability abnormal", f"{p_abn:.2f}")

            st.caption(
                "This app is for educational/screening purposes only."
            )


if __name__ == "__main__":
    main()
