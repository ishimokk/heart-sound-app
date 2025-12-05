import streamlit as st
import numpy as np
import io

# Optional libraries for real audio processing / model
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
CLIP_SECONDS = 5  # seconds


@st.cache_resource(show_spinner=False)
def load_model(path: str = "heart_cnn_model.h5"):
    """
    Load your trained CNN model if it exists, otherwise return None.
    On Streamlit Cloud you probably do NOT have this file yet,
    so the app will usually run in demo mode.
    """
    if tf is None:
        return None
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        return None


def load_and_fix_audio(file) -> np.ndarray:
    """
    Load audio and trim/pad to a fixed length.
    If librosa is missing, just make low-amplitude noise (demo mode).
    """
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
    """
    Turn 1D audio into a spectrogram for the CNN.
    If librosa is missing, use a simple FFT-based representation.
    """
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
    # shape: (1, freq, time, 1)
    return S_norm[np.newaxis, ..., np.newaxis].astype("float32")


def compute_risk_score(answers: dict) -> float:
    """
    Very simple risk scoring based on questionnaire answers.
    This is NOT a medical risk calculator – just a demo to combine
    history + audio in a more realistic way.
    Returns a value between 0 and 1 (higher = more risk factors).
    """
    score = 0
    max_score = 0

    # Age
    age = answers["age"]
    if age == "Under 30":
        score += 0
        max_score += 2
    elif age == "30–44":
        score += 1
        max_score += 2
    elif age == "45–59":
        score += 1.5
        max_score += 2
    else:  # 60+
        score += 2
        max_score += 2

    # Sex at birth (slight weight, just demo)
    sex = answers["sex"]
    max_score += 1
    if sex == "Male":
        score += 0.5
    elif sex == "Female":
        score += 0.3
    else:
        score += 0.4  # other / prefer not to say

    # Simple yes/no factors (1 point each)
    yes_no_keys = [
        "high_bp",
        "high_chol",
        "diabetes",
        "smoker_now",
        "smoker_past",
        "family_early_heart",
        "known_heart_disease",
        "known_valve_disease",
        "prior_heart_surgery",
        "kidney_disease",
        "sleep_apnea",
        "sedentary",
        "high_weight",
        "uses_stimulants",
        "high_alcohol",
    ]
    for k in yes_no_keys:
        max_score += 1
        if answers[k]:
            score += 1

    # Symptom-related (slightly higher weight)
    symptom_keys = [
        "chest_pain",
        "short_of_breath",
        "palpitations",
        "fainting",
        "leg_swelling",
    ]
    for k in symptom_keys:
        max_score += 1.5
        if answers[k]:
            score += 1.5

    # Normalize to 0–1
    if max_score == 0:
        return 0.0
    return float(np.clip(score / max_score, 0.0, 1.0))


def audio_abnormality_score(spec: np.ndarray) -> float:
    """
    Simple 'murmur' score based on high-frequency energy in the spectrogram.
    0 = looks normal, 1 = lots of high-frequency energy.
    """
    spec2d = spec[0, :, :, 0]  # (freq, time)
    total_energy = float(np.sum(spec2d) + 1e-8)
    # treat top ~40% of mel bands as "high frequency"
    split = int(spec2d.shape[0] * 0.6)
    high_energy = float(np.sum(spec2d[split:, :]))
    murmur_score = high_energy / total_energy  # 0–1

    # Map murmur_score into 0–1 with a threshold window
    raw_abn = (murmur_score - 0.3) / 0.2  # around 0.3–0.5
    return float(np.clip(raw_abn, 0.0, 1.0))


def model_predict(model, spec: np.ndarray, risk_score: float):
    """
    If a real CNN exists, use it for audio.
    In demo mode, combine:
      - audio abnormality score (from spectrogram)
      - questionnaire-based risk score
    """
    if model is not None:
        probs = model.predict(spec, verbose=0)[0]
        p_normal_audio = float(probs[0])
        p_abnormal_audio = float(probs[1]) if len(probs) > 1 else 1.0 - p_normal_audio
    else:
        # DEMO AUDIO SCORE: based on murmur-like high-frequency energy
        p_abnormal_audio = audio_abnormality_score(spec)
        p_normal_audio = 1.0 - p_abnormal_audio

    # Combine audio + questionnaire (weights can be tuned)
    # Here: 60% audio, 40% questionnaire
    p_abn_combined = 0.6 * p_abnormal_audio + 0.4 * risk_score
    p_abn_combined = float(np.clip(p_abn_combined, 0.0, 1.0))
    p_norm_combined = 1.0 - p_abn_combined

    label = "Likely NORMAL" if p_norm_combined >= p_abn_combined else "Possible ABNORMAL"
    return label, p_norm_combined, p_abn_combined, p_abnormal_audio


def main():
    st.title("🫀 Heart Sound Screening Prototype")
    st.write(
        "This app is a **research/education prototype** that combines a recorded heart sound "
        "with a simple risk questionnaire to flag patterns that *might* be consistent with "
        "valve problems or other heart issues.\n\n"
        "**It is NOT a medical device and does NOT give a diagnosis.**"
    )

    with st.expander("How this demo works", expanded=False):
        st.markdown(
            """
- You answer a short questionnaire about medical history and symptoms.
- You upload a short heart sound recording (e.g., from a digital stethoscope).
- The app converts the sound to a spectrogram and computes a simple “murmur-like” score.
- In demo mode (no trained model file), it combines:
  - the **audio-based score** and
  - the **questionnaire-based risk score**
- The output is a **screening-style label**: `Likely NORMAL` or `Possible ABNORMAL`.\n
This is just a prototype and should **never** be used to make medical decisions.
"""
        )

    # ---------------------------
    # Questionnaire (20+ items)
    # ---------------------------
    st.subheader("Step 1: Answer a short heart-risk questionnaire")

    col1, col2 = st.columns(2)

    with col1:
        age = st.selectbox(
            "Age group",
            ["Under 30", "30–44", "45–59", "60+"],
        )
        sex = st.selectbox(
            "Sex assigned at birth",
            ["Male", "Female", "Other / Prefer not to say"],
        )
        high_bp = st.checkbox("Have you ever been told you have high blood pressure?")
        high_chol = st.checkbox("High cholesterol (or on cholesterol medicine)?")
        diabetes = st.checkbox("Diabetes or prediabetes?")
        smoker_now = st.checkbox("Do you currently smoke or vape nicotine?")
        smoker_past = st.checkbox("Did you regularly smoke in the past?")
        family_early_heart = st.checkbox(
            "Family history of heart attack or serious heart disease before age 55 (men) or 65 (women)?"
        )
        known_heart_disease = st.checkbox(
            "Have you ever been told you have heart disease (e.g., coronary artery disease)?"
        )
        known_valve_disease = st.checkbox(
            "Have you ever been told you have a valve problem or heart murmur?"
        )

    with col2:
        prior_heart_surgery = st.checkbox("Have you ever had heart surgery or a stent?")
        kidney_disease = st.checkbox("Chronic kidney disease?")
        sleep_apnea = st.checkbox("Sleep apnea (or use a CPAP machine)?")
        high_weight = st.checkbox("Are you significantly overweight (BMI in obese range)?")
        sedentary = st.checkbox("Do you exercise less than 1–2 times per week?")
        uses_stimulants = st.checkbox(
            "Do you regularly use stimulant drugs (e.g., ADHD meds, illicit stimulants)?"
        )
        high_alcohol = st.checkbox("Do you drink alcohol heavily (most days of the week)?")
        chest_pain = st.checkbox(
            "Do you get chest pain or pressure, especially with activity?"
        )
        short_of_breath = st.checkbox(
            "Do you get short of breath easily, more than people your age?"
        )
        palpitations = st.checkbox("Do you feel your heart racing, skipping beats, or pounding?")
        fainting = st.checkbox("Have you had fainting or near-fainting spells?")
        leg_swelling = st.checkbox("Do you often have swelling in your legs or ankles?")

    answers = {
        "age": age,
        "sex": sex,
        "high_bp": high_bp,
        "high_chol": high_chol,
        "diabetes": diabetes,
        "smoker_now": smoker_now,
        "smoker_past": smoker_past,
        "family_early_heart": family_early_heart,
        "known_heart_disease": known_heart_disease,
        "known_valve_disease": known_valve_disease,
        "prior_heart_surgery": prior_heart_surgery,
        "kidney_disease": kidney_disease,
        "sleep_apnea": sleep_apnea,
        "high_weight": high_weight,
        "sedentary": sedentary,
        "uses_stimulants": uses_stimulants,
        "high_alcohol": high_alcohol,
        "chest_pain": chest_pain,
        "short_of_breath": short_of_breath,
        "palpitations": palpitations,
        "fainting": fainting,
        "leg_swelling": leg_swelling,
    }

    risk_score = compute_risk_score(answers)
    st.write(
        f"**Questionnaire risk index (0–1):** `{risk_score:.2f}` "
        "(higher means more risk factors, not a diagnosis)"
    )

    # ---------------------------
    # Audio upload + analysis
    # ---------------------------
    st.subheader("Step 2: Upload a heart sound recording")

    audio_file = st.file_uploader(
        "Upload a short heart sound recording (.wav recommended)",
        type=["wav", "mp3", "ogg"],
    )

    model = load_model()

    if model is None:
        st.info(
            "No trained CNN model file (`heart_cnn_model.h5`) was found, so the app is running "
            "in **demo mode**. The result is based on simple patterns in the sound plus your "
            "risk questionnaire, and is **not** a real medical diagnosis."
        )

    if audio_file is not None:
        st.audio(audio_file, format="audio/wav")

        if st.button("Analyze heart sound"):
            with st.spinner("Analyzing..."):
                y = load_and_fix_audio(audio_file)
                spec = make_spectrogram(y)
                label, p_norm, p_abn, p_abn_audio = model_predict(
                    model, spec, risk_score=risk_score
                )

            st.subheader("Result")
            st.markdown(f"### **{label}**")
            st.write(f"- **Combined probability abnormal:** `{p_abn:.2f}`")
            st.write(f"- **Audio-based abnormality score:** `{p_abn_audio:.2f}`")
            st.write(f"- **Questionnaire risk index:** `{risk_score:.2f}`")

            st.caption(
                "This app is a prototype for educational and research purposes only. "
                "It cannot diagnose or rule out any heart condition. If you have symptoms "
                "or concerns, you should talk to a licensed healthcare professional."
            )


if __name__ == "__main__":
    main()
