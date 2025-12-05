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

st.set_page_config(
    page_title="Heart Sound Screening Prototype",
    page_icon="🫀",
    layout="centered",
)

TOTAL_PAGES = 5

# ----------------- LIGHT STYLING -----------------
st.markdown(
    """
    <style>
        /* Page background */
        .stApp {
            background-color: #f5f7fb;
        }
        /* Cards */
        .card {
            background-color: #ffffff;
            padding: 1.3rem 1.6rem;
            border-radius: 0.9rem;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
            margin-bottom: 1.2rem;
        }
        .card h3 {
            margin-top: 0;
            margin-bottom: 0.6rem;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .muted {
            color: #6b7280;
            font-size: 0.9rem;
        }
        .pill {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            background-color: #eef2ff;
            color: #4338ca;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def card(title: str, body: str):
    st.markdown(
        f"""
        <div class="card">
            <div class="pill">{title}</div>
            <div class="muted">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ----------------- GLOBAL QUESTIONS -----------------
HISTORY_QUESTIONS = {
    "high_bp": "Have you ever been told you have high blood pressure?",
    "high_chol": "Have you ever been told you have high cholesterol or been prescribed cholesterol medication?",
    "diabetes": "Have you ever been diagnosed with diabetes or prediabetes?",
    "smoker_now": "Do you currently smoke cigarettes, cigars, or vape nicotine regularly?",
    "smoker_past": "Have you regularly smoked in the past for more than one year?",
    "family_early_heart": "Do you have a family history of heart attack or serious heart disease at a young age (before 55 in men or 65 in women)?",
    "known_heart_disease": "Have you ever been told that you have heart disease, such as coronary artery disease or a previous heart attack?",
    "known_valve_disease": "Have you ever been told that you have a heart valve problem or a heart murmur?",
    "prior_heart_surgery": "Have you ever had heart surgery, a stent, or another procedure on your heart?",
    "kidney_disease": "Have you ever been told that you have chronic kidney disease?",
    "sleep_apnea": "Have you been diagnosed with sleep apnea or do you use a CPAP machine at night?",
    "high_weight": "Has a healthcare provider ever told you that your weight puts you in the obese range or at increased risk for heart problems?",
    "sedentary": "Do you exercise less than one to two times per week on average?",
    "uses_stimulants": "Do you regularly use stimulant medications or stimulant drugs (for example, ADHD medications or illicit stimulants)?",
    "high_alcohol": "Do you drink alcohol heavily on most days of the week?",
}

SYMPTOM_QUESTIONS = {
    "short_of_breath": "Do you experience shortness of breath, especially during or after physical activity?",
    "low_stamina": "Do you feel unusually tired or have reduced stamina compared to your usual activity level?",
    "dizzy_faint": "Do you feel dizzy, lightheaded, or have you fainted recently?",
    "chest_pain": "Have you experienced chest pain or pressure, especially with activity or stress?",
    "palpitations": "Do you feel a rapid or irregular heartbeat, like palpitations or fluttering in your chest?",
    "leg_swelling": "Have you noticed swelling in your ankles, feet, or lower legs?",
    "cough": "Do you have a persistent cough that is not explained by a cold or allergies?",
    "fever": "Do you currently have a fever or have you had a recent fever that could indicate an infection?",
}

# ----------------- SESSION STATE -----------------
if "page" not in st.session_state:
    st.session_state.page = 1

if "answers" not in st.session_state:
    st.session_state.answers = {}

if "result" not in st.session_state:
    st.session_state.result = None  # will store last analysis


# ----------------- MODEL LOADING -----------------
@st.cache_resource
def load_model(path="heart_cnn_model.h5"):
    if tf is None:
        return None
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        return None


# ----------------- AUDIO HELPERS -----------------
def load_and_fix_audio(file):
    sample_rate = 16000
    clip_seconds = 5
    target_len = sample_rate * clip_seconds

    if librosa is None:
        return np.random.randn(target_len).astype("float32") * 0.01

    y, sr = librosa.load(file, sr=sample_rate, mono=True)
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
        y=y,
        sr=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=64,
        window="hann",
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    return S_norm[np.newaxis, ..., np.newaxis].astype("float32")


# ----------------- RISK SCORING -----------------
def compute_risk_score(a: dict) -> float:
    score = 0.0
    max_score = 0.0

    # Age
    age = a.get("age", "Under 30")
    age_weights = {
        "Under 30": 0.0,
        "30–44": 1.0,
        "45–59": 1.5,
        "60+": 2.0,
    }
    score += age_weights.get(age, 0.0)
    max_score += 2.0

    # Sex at birth (tiny weight, just demo)
    sex = a.get("sex", "Other / Prefer not to say")
    max_score += 1.0
    if sex == "Male":
        score += 0.5
    elif sex == "Female":
        score += 0.3
    else:
        score += 0.4

    # History
    for key in HISTORY_QUESTIONS.keys():
        max_score += 1.0
        if a.get(key, False):
            score += 1.0

    # Symptoms
    for key in SYMPTOM_QUESTIONS.keys():
        max_score += 1.5
        if a.get(key, False):
            score += 1.5

    if max_score == 0:
        return 0.0
    return float(np.clip(score / max_score, 0.0, 1.0))


# ----------------- AUDIO ABNORMALITY SCORE -----------------
def audio_abnormality_score(spec) -> float:
    spec2d = spec[0, :, :, 0]
    total_energy = float(np.sum(spec2d) + 1e-8)
    split = int(spec2d.shape[0] * 0.6)
    high_energy = float(np.sum(spec2d[split:, :]))
    murmur_score = high_energy / total_energy
    raw = (murmur_score - 0.3) / 0.2
    return float(np.clip(raw, 0.0, 1.0))


# ----------------- MODEL PREDICTION -----------------
def model_predict(model, spec, risk_score: float):
    if model is not None:
        probs = model.predict(spec, verbose=0)[0]
        p_normal_audio = float(probs[0])
        p_abnormal_audio = float(probs[1]) if len(probs) > 1 else 1.0 - p_normal_audio
    else:
        p_abnormal_audio = audio_abnormality_score(spec)
        p_normal_audio = 1.0 - p_abnormal_audio

    p_abn_combined = float(np.clip(0.6 * p_abnormal_audio + 0.4 * risk_score, 0.0, 1.0))
    p_norm_combined = 1.0 - p_abn_combined

    label = "Likely NORMAL" if p_norm_combined >= p_abn_combined else "Possible ABNORMAL"

    return {
        "label": label,
        "p_norm": p_norm_combined,
        "p_abn": p_abn_combined,
        "p_abn_audio": p_abnormal_audio,
        "risk_score": risk_score,
    }


# ----------------- PAGE NAV HELPERS -----------------
def next_page():
    if st.session_state.page < TOTAL_PAGES:
        st.session_state.page += 1


def prev_page():
    if st.session_state.page > 1:
        st.session_state.page -= 1


# ----------------- HEADER + PROGRESS -----------------
st.markdown(
    """
    <div style="text-align:center; margin-bottom: 0.5rem;">
        <h1 style="margin-bottom:0.2rem;">🫀 Heart Sound Screening Prototype</h1>
        <p class="muted" style="margin-top:0;">
            A research and education prototype that combines a heart sound recording with a brief risk questionnaire.
            This is <strong>not</strong> a medical device and does <strong>not</strong> give a diagnosis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("How this prototype works"):
    st.markdown(
        """
- You answer a brief questionnaire about your health history and symptoms.
- You upload a short heart sound recording (for example, from a digital stethoscope).
- The app creates a spectrogram of the sound and estimates how “murmur-like” it looks.
- It then combines the audio-based estimate with the questionnaire-based risk index.\n
This is only a prototype and should **never** be used for real medical decisions.
"""
    )

current_page = st.session_state.page
st.write(f"**Step {current_page} of {TOTAL_PAGES}**")
st.progress(current_page / TOTAL_PAGES)

st.markdown("<br>", unsafe_allow_html=True)

# ===========================
#          PAGE 1
# ===========================
if st.session_state.page == 1:
    st.subheader("Step 1 · Basic information")

    with st.container():
        st.session_state.answers["age"] = st.selectbox(
            "What is your age group?",
            ["Under 30", "30–44", "45–59", "60+"],
        )
        st.session_state.answers["sex"] = st.selectbox(
            "What sex were you assigned at birth?",
            ["Male", "Female", "Other / Prefer not to say"],
        )

    st.button("Next ➜", on_click=next_page)


# ===========================
#          PAGE 2
# ===========================
elif st.session_state.page == 2:
    st.subheader("Step 2 · Medical history")

    card(
        "About these questions",
        "These questions ask about long-term health conditions and lifestyle factors that may influence heart risk. "
        "Answer as honestly as you can. You can leave questions unchecked if they do not apply.",
    )

    for key, question in HISTORY_QUESTIONS.items():
        st.session_state.answers[key] = st.checkbox(
            question,
            value=st.session_state.answers.get(key, False),
        )

    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅ Back", on_click=prev_page)
    with col2:
        st.button("Next ➜", on_click=next_page)


# ===========================
#          PAGE 3
# ===========================
elif st.session_state.page == 3:
    st.subheader("Step 3 · Current symptoms")

    card(
        "About these questions",
        "These questions focus on how you have been feeling recently. "
        "They help flag symptoms that sometimes appear in people with heart or valve problems.",
    )

    for key, question in SYMPTOM_QUESTIONS.items():
        st.session_state.answers[key] = st.checkbox(
            question,
            value=st.session_state.answers.get(key, False),
        )

    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅ Back", on_click=prev_page)
    with col2:
        st.button("Next ➜", on_click=next_page)


# ===========================
#          PAGE 4
# ===========================
elif st.session_state.page == 4:
    st.subheader("Step 4 · Upload heart sound and analyze")

    risk_score = compute_risk_score(st.session_state.answers)

    card(
        "Questionnaire risk index",
        f"Based on your answers so far, the questionnaire risk index is **{risk_score:.2f}** on a 0–1 scale. "
        "Higher values mean more reported risk factors and symptoms, but this is not a diagnosis.",
    )

    audio_file = st.file_uploader(
        "Upload a short heart sound recording (.wav is recommended)",
        type=["wav", "mp3", "ogg"],
    )

    model = load_model()
    if model is None:
        card(
            "Model status",
            "No trained heart sound model file (`heart_cnn_model.h5`) was found, so the audio analysis uses a "
            "simple pattern-based estimate. This is for demonstration only and is **not** a medical test.",
        )

    if audio_file is not None:
        st.audio(audio_file)

        if st.button("Analyze heart sound"):
            y = load_and_fix_audio(audio_file)
            spec = make_spectrogram(y)
            result = model_predict(model, spec, risk_score)
            st.session_state.result = result

            st.markdown(
                """
                <div class="card">
                    <div class="pill">Screening-style result</div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f"#### {result['label']}")
            st.markdown(
                f"- Audio-based abnormality score: `{result['p_abn_audio']:.2f}`  \n"
                f"- Questionnaire risk index: `{result['risk_score']:.2f}`  \n"
                f"- Combined probability of abnormal finding: `{result['p_abn']:.2f}`"
            )
            st.markdown(
                """
                <p class="muted">
                This result is part of a research and education prototype. It is not a diagnosis and cannot
                confirm or rule out any medical condition. If you have symptoms or concerns, you should speak
                with a licensed healthcare professional.
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅ Back", on_click=prev_page)
    with col2:
        st.button("Go to summary ➜", on_click=next_page)


# ===========================
#          PAGE 5
# ===========================
elif st.session_state.page == 5:
    st.subheader("Step 5 · Summary of your answers and result")

    answers = st.session_state.answers
    result = st.session_state.result

    st.markdown("### Basic information")
    st.write(f"- Age group: **{answers.get('age', 'Not provided')}**")
    st.write(f"- Sex assigned at birth: **{answers.get('sex', 'Not provided')}**")

    st.markdown("---")
    st.markdown("### Medical history")
    for key, question in HISTORY_QUESTIONS.items():
        value = answers.get(key, False)
        st.write(f"- {question} — **{'Yes' if value else 'No'}**")

    st.markdown("---")
    st.markdown("### Symptoms")
    for key, question in SYMPTOM_QUESTIONS.items():
        value = answers.get(key, False)
        st.write(f"- {question} — **{'Yes' if value else 'No'}**")

    st.markdown("---")
    st.markdown("### Screening-style result")

    if result is None:
        st.warning(
            "No heart sound has been analyzed yet in this session. "
            "Please go back to Step 4, upload a recording, and run the analysis."
        )
    else:
        st.markdown(f"#### {result['label']}")
        st.write(f"- Audio-based abnormality score: `{result['p_abn_audio']:.2f}`")
        st.write(f"- Questionnaire risk index: `{result['risk_score']:.2f}`")
        st.write(f"- Combined probability of abnormal finding: `{result['p_abn']:.2f}`")
        st.caption(
            "This summary is part of a research prototype and is not a diagnosis. "
            "It cannot confirm or rule out any heart condition."
        )

    st.button("⬅ Back", on_click=prev_page)
