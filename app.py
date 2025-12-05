import streamlit as st
import numpy as np

# Optional imports for real audio processing
try:
    import librosa
except ImportError:
    librosa = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

st.set_page_config(page_title="Heart Sound Screening Prototype", page_icon="🫀", layout="centered")

# Initialize page state
if "page" not in st.session_state:
    st.session_state.page = 1

# Initialize questionnaire answers
if "answers" not in st.session_state:
    st.session_state.answers = {}

# ---- LOAD MODEL ----
@st.cache_resource
def load_model(path="heart_cnn_model.h5"):
    if tf is None:
        return None
    try:
        return tf.keras.models.load_model(path)
    except:
        return None


# ---- AUDIO FUNCTIONS ----
def load_and_fix_audio(file):
    target_len = 16000 * 5
    if librosa is None:
        return np.random.randn(target_len).astype("float32") * 0.01
    y, sr = librosa.load(file, sr=16000, mono=True)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    return y


def make_spectrogram(y):
    if librosa is None:
        spec = np.abs(np.fft.rfft(y))[:512]
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)
        return spec.reshape(1, -1, 1, 1).astype("float32")
    S = librosa.feature.melspectrogram(
        y=y, sr=16000, n_fft=1024, hop_length=256, n_mels=64, window="hann"
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_norm[np.newaxis, ..., np.newaxis].astype("float32")


# ---- QUESTIONNAIRE RISK SCORE ----
def compute_risk_score(a):
    score, max_score = 0, 0

    # Age
    age_weight = {"Under 30": 0, "30–44": 1, "45–59": 1.5, "60+": 2}
    score += age_weight[a["age"]]
    max_score += 2

    # Sex minimal weight
    score += 0.5 if a["sex"] == "Male" else 0.3
    max_score += 1

    # Yes/no risk factors
    yes_no_keys = [
        "high_bp", "high_chol", "diabetes", "smoker_now", "smoker_past",
        "family_early_heart", "known_heart_disease", "known_valve_disease",
        "prior_heart_surgery", "kidney_disease", "sleep_apnea",
        "high_weight", "sedentary", "uses_stimulants", "high_alcohol"
    ]
    for k in yes_no_keys:
        score += 1 if a[k] else 0
        max_score += 1

    symptom_keys = ["chest_pain", "short_of_breath", "palpitations", "fainting", "leg_swelling"]
    for k in symptom_keys:
        score += 1.5 if a[k] else 0
        max_score += 1.5

    return float(np.clip(score / max_score, 0, 1))


# ---- AUDIO SCORE ----
def audio_abnormality_score(spec):
    spec2d = spec[0, :, :, 0]
    total_energy = float(np.sum(spec2d) + 1e-8)
    split = int(spec2d.shape[0] * 0.6)
    high_energy = float(np.sum(spec2d[split:, :]))
    murmur_score = high_energy / total_energy
    raw = (murmur_score - 0.3) / 0.2
    return float(np.clip(raw, 0, 1))


# ---- MODEL PREDICT ----
def model_predict(model, spec, risk):
    if model is not None:
        probs = model.predict(spec, verbose=0)[0]
        p_norm = float(probs[0])
        p_abn_audio = float(probs[1])
    else:
        p_abn_audio = audio_abnormality_score(spec)
        p_norm = 1 - p_abn_audio

    # Combine audio + questionnaire
    p_abn_final = float(np.clip(0.6 * p_abn_audio + 0.4 * risk, 0, 1))
    p_norm_final = 1 - p_abn_final

    label = "Likely NORMAL" if p_norm_final >= p_abn_final else "Possible ABNORMAL"
    return label, p_norm_final, p_abn_final, p_abn_audio


# ---- PAGE NAVIGATION ----
def next_page():
    st.session_state.page += 1


def prev_page():
    st.session_state.page -= 1


# ===================================
#           PAGE 1
# ===================================
if st.session_state.page == 1:
    st.header("🩺 Step 1: Basic Information")

    st.session_state.answers["age"] = st.selectbox("Age group", ["Under 30", "30–44", "45–59", "60+"])
    st.session_state.answers["sex"] = st.selectbox("Sex assigned at birth", ["Male", "Female", "Other"])

    st.button("Next ➜", on_click=next_page)


# ===================================
#           PAGE 2
# ===================================
elif st.session_state.page == 2:
    st.header("🧬 Step 2: Medical History")

    for q in [
        "high_bp", "high_chol", "diabetes", "smoker_now", "smoker_past",
        "family_early_heart", "known_heart_disease", "known_valve_disease",
        "prior_heart_surgery", "kidney_disease", "sleep_apnea",
        "high_weight", "sedentary", "uses_stimulants", "high_alcohol"
    ]:
        st.session_state.answers[q] = st.checkbox(q.replace("_", " ").title())

    col1, col2 = st.columns(2)
    col1.button("⬅ Back", on_click=prev_page)
    col2.button("Next ➜", on_click=next_page)


# ===================================
#           PAGE 3
# ===================================
elif st.session_state.page == 3:
    st.header("❤️ Step 3: Symptoms")

    for q in ["chest_pain", "short_of_breath", "palpitations", "fainting", "leg_swelling"]:
        st.session_state.answers[q] = st.checkbox(q.replace("_", " ").title())

    col1, col2 = st.columns(2)
    col1.button("⬅ Back", on_click=prev_page)
    col2.button("Next ➜", on_click=next_page)


# ===================================
#           PAGE 4
# ===================================
elif st.session_state.page == 4:
    st.header("🎧 Step 4: Upload Heart Sound")

    risk_score = compute_risk_score(st.session_state.answers)
    st.info(f"Your questionnaire risk index: **{risk_score:.2f}** (0–1)")

    audio_file = st.file_uploader("Upload heart sound (.wav recommended)", type=["wav", "mp3", "ogg"])

    model = load_model()

    if audio_file is not None:
        st.audio(audio_file)

        if st.button("Analyze"):
            y = load_and_fix_audio(audio_file)
            spec = make_spectrogram(y)
            label, p_norm, p_abn, p_abn_audio = model_predict(model, spec, risk_score)

            st.subheader("Result")
            st.write(f"### **{label}**")
            st.write(f"Audio abnormality: `{p_abn_audio:.2f}`")
            st.write(f"Questionnaire risk: `{risk_score:.2f}`")
            st.write(f"Combined abnormal probability: **{p_abn:.2f}**")

    st.button("⬅ Back", on_click=prev_page)
