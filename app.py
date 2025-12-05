import streamlit as st
import numpy as np

# Optional imports for real audio processing / model
try:
    import librosa
except ImportError:
    librosa = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

# ------------ CONFIG ------------
st.set_page_config(
    page_title="Heart Sound Screening Prototype",
    page_icon="🫀",
    layout="centered",
)

TOTAL_PAGES = 9  # we now have 9 steps

# ------------ BASIC STYLING (HIGH CONTRAST, MEDICAL STYLE) ------------
st.markdown(
    """
    <style>
        /* Force dark text everywhere for readability */
        body, .stApp, .stMarkdown, .stMarkdown p, .stMarkdown li,
        h1, h2, h3, h4, h5, h6,
        p, label, span, div {
            color: #111827 !important;
        }

        /* Page background */
        .stApp {
            background-color: #ffffff !important;
        }

        /* Header styling */
        .app-header {
            text-align: center;
            margin-bottom: 0.75rem;
        }
        .app-header h1 {
            margin-bottom: 0.25rem;
        }
        .app-header p {
            margin-top: 0;
            font-size: 0.95rem;
        }

        /* Card styling */
        .card {
            background-color: #ffffff;
            padding: 1.1rem 1.3rem;
            border-radius: 0.75rem;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 8px rgba(15,23,42,0.05);
            margin-bottom: 0.8rem;
        }
        .card-title {
            font-weight: 600;
            margin-bottom: 0.35rem;
        }
        .card-text {
            font-size: 0.95rem;
        }

        /* Footer */
        .footer {
            margin-top: 2rem;
            padding-top: 0.75rem;
            border-top: 1px solid #e5e7eb;
            font-size: 0.9rem;
            text-align: center;
        }
        .footer span {
            font-weight: 600;
        }

        /* Links (if any) */
        a {
            color: #0A74DA !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)



def card(title: str, body: str):
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <div class="card-text">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------ QUESTION DEFINITIONS ------------

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

# Lifestyle questions with discrete options
LIFESTYLE_QUESTIONS = {
    "exercise_level": (
        "How often do you exercise in a typical week?",
        [
            "Rarely or never",
            "1–2 days per week",
            "3–4 days per week",
            "5 or more days per week",
        ],
    ),
    "diet_quality": (
        "How would you describe your usual diet?",
        [
            "Mostly fast food or processed foods",
            "Mixed (some healthy, some unhealthy)",
            "Mostly home cooked with fruits and vegetables",
        ],
    ),
    "sleep_hours": (
        "On average, how many hours do you sleep per night?",
        [
            "Less than 5 hours",
            "5–6 hours",
            "7–8 hours",
            "More than 8 hours",
        ],
    ),
    "stress_level": (
        "How would you rate your overall stress level?",
        [
            "Very high",
            "High",
            "Moderate",
            "Low",
        ],
    ),
    "caffeine_intake": (
        "How much caffeine do you usually have in a day? (coffee, tea, energy drinks, soda)",
        [
            "None or very little",
            "1–2 caffeinated drinks",
            "3–4 caffeinated drinks",
            "More than 4 caffeinated drinks",
        ],
    ),
    "water_intake": (
        "How many cups or glasses of water do you usually drink per day?",
        [
            "Fewer than 3 cups",
            "3–5 cups",
            "6–8 cups",
            "More than 8 cups",
        ],
    ),
}

# Vitals fields (optional)
VITAL_FIELDS = {
    "resting_hr": "Resting heart rate (beats per minute)",
    "systolic_bp": "Systolic blood pressure (top number, mmHg)",
    "diastolic_bp": "Diastolic blood pressure (bottom number, mmHg)",
    "spo2": "Oxygen saturation (SpO₂ percentage, if known)",
    "temperature": "Body temperature (in °F or °C, whichever you prefer)",
}

# ------------ SESSION STATE ------------
if "page" not in st.session_state:
    st.session_state.page = 1

if "answers" not in st.session_state:
    st.session_state.answers = {}

if "result" not in st.session_state:
    st.session_state.result = None

if "audio_uploaded" not in st.session_state:
    st.session_state.audio_uploaded = False


# ------------ MODEL LOADING ------------
@st.cache_resource
def load_model(path: str = "heart_cnn_model.h5"):
    if tf is None:
        return None
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        return None


# ------------ AUDIO HELPERS ------------
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


# ------------ RISK SCORING ------------
def compute_bmi(weight_kg: float, height_cm: float):
    if weight_kg is None or height_cm is None or height_cm <= 0:
        return None
    h_m = height_cm / 100.0
    bmi = weight_kg / (h_m * h_m)
    return float(bmi)


def compute_history_symptom_risk(a: dict) -> float:
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

    # Sex at birth (tiny weight)
    sex = a.get("sex", "Other / Prefer not to say")
    max_score += 1.0
    if sex == "Male":
        score += 0.5
    elif sex == "Female":
        score += 0.3
    else:
        score += 0.4

    # History yes/no
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


def compute_lifestyle_score(a: dict) -> float:
    # 0 = healthier, 1 = less healthy
    score = 0.0
    max_score = 0.0

    # Exercise
    ex = a.get("exercise_level", "Rarely or never")
    max_score += 1.0
    if ex == "Rarely or never":
        score += 1.0
    elif ex == "1–2 days per week":
        score += 0.7
    elif ex == "3–4 days per week":
        score += 0.3
    else:  # 5 or more days
        score += 0.1

    # Diet
    diet = a.get("diet_quality", "Mixed (some healthy, some unhealthy)")
    max_score += 1.0
    if diet == "Mostly fast food or processed foods":
        score += 1.0
    elif diet == "Mixed (some healthy, some unhealthy)":
        score += 0.6
    else:
        score += 0.2

    # Sleep
    sleep = a.get("sleep_hours", "5–6 hours")
    max_score += 1.0
    if sleep == "Less than 5 hours":
        score += 1.0
    elif sleep == "5–6 hours":
        score += 0.7
    elif sleep == "7–8 hours":
        score += 0.2
    else:
        score += 0.3

    # Stress
    stress = a.get("stress_level", "Moderate")
    max_score += 1.0
    if stress == "Very high":
        score += 1.0
    elif stress == "High":
        score += 0.8
    elif stress == "Moderate":
        score += 0.5
    else:
        score += 0.2

    # Caffeine
    caf = a.get("caffeine_intake", "1–2 caffeinated drinks")
    max_score += 1.0
    if caf == "None or very little":
        score += 0.2
    elif caf == "1–2 caffeinated drinks":
        score += 0.4
    elif caf == "3–4 caffeinated drinks":
        score += 0.7
    else:
        score += 1.0

    # Water
    water = a.get("water_intake", "3–5 cups")
    max_score += 1.0
    if water == "Fewer than 3 cups":
        score += 1.0
    elif water == "3–5 cups":
        score += 0.7
    elif water == "6–8 cups":
        score += 0.3
    else:
        score += 0.2

    if max_score == 0:
        return 0.0
    return float(np.clip(score / max_score, 0.0, 1.0))


def compute_vitals_score(a: dict) -> float:
    """
    Simple scoring from available vitals (if provided).
    Higher = more concerning.
    """
    score = 0.0
    max_score = 0.0

    hr = a.get("resting_hr")
    if hr is not None:
        max_score += 1.0
        if hr < 50 or hr > 100:
            score += 1.0
        elif 90 < hr <= 100 or 50 <= hr < 60:
            score += 0.6
        else:
            score += 0.2

    sbp = a.get("systolic_bp")
    dbp = a.get("diastolic_bp")
    if sbp is not None and dbp is not None:
        max_score += 1.0
        if sbp >= 140 or dbp >= 90:
            score += 1.0
        elif sbp >= 130 or dbp >= 85:
            score += 0.7
        elif sbp < 90 or dbp < 60:
            score += 0.8
        else:
            score += 0.3

    spo2 = a.get("spo2")
    if spo2 is not None:
        max_score += 1.0
        if spo2 < 90:
            score += 1.0
        elif spo2 < 95:
            score += 0.7
        else:
            score += 0.2

    temp = a.get("temperature")
    if temp is not None:
        max_score += 1.0
        # assume °F or °C; just look for fever or low temp
        if temp >= 100.4 or temp <= 95:
            score += 1.0
        elif 99.0 <= temp < 100.4:
            score += 0.6
        else:
            score += 0.2

    if max_score == 0:
        return 0.0
    return float(np.clip(score / max_score, 0.0, 1.0))


def audio_abnormality_score(spec) -> float:
    spec2d = spec[0, :, :, 0]
    total_energy = float(np.sum(spec2d) + 1e-8)
    split = int(spec2d.shape[0] * 0.6)
    high_energy = float(np.sum(spec2d[split:, :]))
    murmur_score = high_energy / total_energy
    raw = (murmur_score - 0.3) / 0.2
    return float(np.clip(raw, 0.0, 1.0))


def model_predict(model, spec, a: dict):
    """
    Combine:
      - Audio-based abnormality
      - History + symptoms
      - Lifestyle
      - Vitals
    """
    history_symptom_risk = compute_history_symptom_risk(a)
    lifestyle_score = compute_lifestyle_score(a)
    vitals_score = compute_vitals_score(a)

    if model is not None:
        probs = model.predict(spec, verbose=0)[0]
        p_normal_audio = float(probs[0])
        p_abnormal_audio = float(probs[1]) if len(probs) > 1 else 1.0 - p_normal_audio
    else:
        p_abnormal_audio = audio_abnormality_score(spec)
        p_normal_audio = 1.0 - p_abnormal_audio

    # Combine non-audio into one risk score
    combined_non_audio = (
        0.5 * history_symptom_risk + 0.3 * lifestyle_score + 0.2 * vitals_score
    )
    combined_non_audio = float(np.clip(combined_non_audio, 0.0, 1.0))

    # Overall
    p_abn = float(np.clip(0.6 * p_abnormal_audio + 0.4 * combined_non_audio, 0.0, 1.0))
    p_norm = 1.0 - p_abn

    label = "Likely NORMAL" if p_norm >= p_abn else "Possible ABNORMAL"

    # Simple severity bucket
    if p_abn < 0.3:
        severity = "Low concern (within this prototype)"
    elif p_abn < 0.6:
        severity = "Moderate concern (within this prototype)"
    else:
        severity = "Higher concern (within this prototype)"

    return {
        "label": label,
        "p_norm": p_norm,
        "p_abn": p_abn,
        "p_abn_audio": p_abnormal_audio,
        "history_symptom_risk": history_symptom_risk,
        "lifestyle_score": lifestyle_score,
        "vitals_score": vitals_score,
        "combined_non_audio": combined_non_audio,
        "severity": severity,
    }


# ------------ NAV HELPERS ------------
def next_page():
    if st.session_state.page < TOTAL_PAGES:
        st.session_state.page += 1


def prev_page():
    if st.session_state.page > 1:
        st.session_state.page -= 1


# ------------ HEADER ------------
st.markdown(
    """
    <div class="app-header">
        <h1>🫀 Heart Sound Screening Prototype</h1>
        <p>
            A research and education prototype that combines a heart sound recording with a brief health questionnaire.
            It is <strong>not</strong> a medical device and does <strong>not</strong> give a diagnosis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

current_page = st.session_state.page
st.write(f"**Step {current_page} of {TOTAL_PAGES}**")
st.progress(current_page / TOTAL_PAGES)
st.write("")

# ------------ PAGE LOGIC ------------

# PAGE 1 – Welcome / Consent
if st.session_state.page == 1:
    st.subheader("Step 1 · Welcome and consent")

    card(
        "What this app does",
        "This app is a prototype that explores how a heart sound recording and simple health questions "
        "might be combined to flag patterns that could be worth checking with a healthcare professional.",
    )
    card(
        "Important limitations",
        "This tool cannot diagnose, treat, or rule out any medical condition. "
        "It has not been tested or approved for clinical use. "
        "If you have chest pain, trouble breathing, fainting, or other serious symptoms, "
        "you should seek immediate medical care.",
    )

    consent = st.checkbox(
        "I understand that this app is for education and research only and is not medical advice."
    )

    if st.button("I understand and want to continue ➜"):
        if consent:
            st.session_state.page = 2
        else:
            st.warning("Please confirm that you understand before continuing.")


# PAGE 2 – Basic information
elif st.session_state.page == 2:
    st.subheader("Step 2 · Basic information")

    answers = st.session_state.answers

    answers["age"] = st.selectbox(
        "What is your age group?",
        ["Under 30", "30–44", "45–59", "60+"],
        index=["Under 30", "30–44", "45–59", "60+"].index(answers.get("age", "Under 30"))
        if "age" in answers
        else 0,
    )

    answers["sex"] = st.selectbox(
        "What sex were you assigned at birth?",
        ["Male", "Female", "Other / Prefer not to say"],
        index=[
            "Male",
            "Female",
            "Other / Prefer not to say",
        ].index(answers.get("sex", "Other / Prefer not to say"))
        if "sex" in answers
        else 2,
    )

    st.write("You can optionally enter your height and weight to estimate body mass index (BMI).")
    col1, col2 = st.columns(2)
    with col1:
        height_cm = st.number_input(
            "Height (in centimeters)",
            min_value=0.0,
            max_value=260.0,
            value=answers.get("height_cm", 0.0),
            step=0.5,
        )
        answers["height_cm"] = height_cm if height_cm > 0 else None
    with col2:
        weight_kg = st.number_input(
            "Weight (in kilograms)",
            min_value=0.0,
            max_value=500.0,
            value=answers.get("weight_kg", 0.0),
            step=0.5,
        )
        answers["weight_kg"] = weight_kg if weight_kg > 0 else None

    bmi = None
    if answers.get("height_cm") is not None and answers.get("weight_kg") is not None:
        bmi = compute_bmi(answers["weight_kg"], answers["height_cm"])
        answers["bmi"] = bmi
        st.write(f"Estimated BMI: **{bmi:.1f}**")
    else:
        answers["bmi"] = None

    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅ Back", on_click=prev_page)
    with col2:
        st.button("Next ➜", on_click=next_page)


# PAGE 3 – Medical history
elif st.session_state.page == 3:
    st.subheader("Step 3 · Medical history")

    card(
        "Long-term conditions",
        "These questions ask about medical conditions that can change heart risk over time. "
        "You can leave items unchecked if they do not apply.",
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


# PAGE 4 – Symptoms
elif st.session_state.page == 4:
    st.subheader("Step 4 · Current symptoms")

    card(
        "How you have been feeling",
        "These questions focus on symptoms. If you have chest pain, difficulty breathing, or fainting, "
        "you should contact a healthcare professional, even if this app shows a low risk.",
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


# PAGE 5 – Lifestyle
elif st.session_state.page == 5:
    st.subheader("Step 5 · Lifestyle")

    card(
        "Daily habits",
        "Lifestyle factors like exercise, diet, sleep, stress, caffeine, and water intake can all influence "
        "overall heart health over time.",
    )

    for key, (question, options) in LIFESTYLE_QUESTIONS.items():
        default_value = st.session_state.answers.get(key, options[0])
        st.session_state.answers[key] = st.selectbox(
            question,
            options,
            index=options.index(default_value) if default_value in options else 0,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅ Back", on_click=prev_page)
    with col2:
        st.button("Next ➜", on_click=next_page)


# PAGE 6 – Vitals
elif st.session_state.page == 6:
    st.subheader("Step 6 · Optional vital signs")

    card(
        "If you know these values",
        "If you have recent measurements from a home blood pressure cuff, pulse oximeter, "
        "smartwatch, or clinic visit, you can enter them here. If not, you can leave them blank.",
    )

    for key, label in VITAL_FIELDS.items():
        existing = st.session_state.answers.get(key, None)
        value = st.number_input(
            label,
            value=float(existing) if isinstance(existing, (int, float)) else 0.0,
            min_value=0.0,
            step=1.0,
        )
        # Treat 0 as "not provided"
        st.session_state.answers[key] = value if value > 0 else None

    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅ Back", on_click=prev_page)
    with col2:
        st.button("Next ➜", on_click=next_page)


# PAGE 7 – Upload heart sound
elif st.session_state.page == 7:
    st.subheader("Step 7 · Upload heart sound")

    card(
        "How to record",
        "Use a digital stethoscope or a phone microphone placed over the chest in a quiet room. "
        "Short recordings of a few seconds focused on heart sounds work best for this prototype.",
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
        st.session_state.audio_uploaded = True
        st.audio(audio_file)

        if st.button("Analyze heart sound ➜"):
            y = load_and_fix_audio(audio_file)
            spec = make_spectrogram(y)
            result = model_predict(model, spec, st.session_state.answers)
            st.session_state.result = result
            st.success("Analysis complete. Go to the next step to see details.")
    else:
        st.session_state.audio_uploaded = False

    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅ Back", on_click=prev_page)
    with col2:
        st.button("Next ➜", on_click=next_page)


# PAGE 8 – Results breakdown
elif st.session_state.page == 8:
    st.subheader("Step 8 · Results overview")

    result = st.session_state.result
    if result is None:
        st.warning(
            "No heart sound analysis is available yet. Please go back to Step 7, upload a recording, and run the analysis."
        )
    else:
        card(
            "Screening-style result",
            f"Overall label: **{result['label']}**\n\n"
            f"Within this prototype, this falls into the category: **{result['severity']}**.",
        )

        st.markdown("#### Numerical scores")
        st.write(f"- Audio-based abnormality score: `{result['p_abn_audio']:.2f}`")
        st.write(f"- History and symptom risk index: `{result['history_symptom_risk']:.2f}`")
        st.write(f"- Lifestyle risk score: `{result['lifestyle_score']:.2f}`")
        st.write(f"- Vitals risk score: `{result['vitals_score']:.2f}`")
        st.write(f"- Combined non-audio risk: `{result['combined_non_audio']:.2f}`")
        st.write(f"- Combined probability of abnormal finding: `{result['p_abn']:.2f}`")

        st.caption(
            "These numbers are part of a research and education prototype. "
            "They do not represent a validated medical risk score."
        )

    col1, col2 = st.columns(2)
    with col1:
        st.button("⬅ Back", on_click=prev_page)
    with col2:
        st.button("Go to full summary ➜", on_click=next_page)


# PAGE 9 – Full summary
elif st.session_state.page == 9:
    st.subheader("Step 9 · Full summary")

    a = st.session_state.answers
    result = st.session_state.result

    st.markdown("### Basic information")
    st.write(f"- Age group: **{a.get('age', 'Not provided')}**")
    st.write(f"- Sex assigned at birth: **{a.get('sex', 'Not provided')}**")

    bmi = a.get("bmi")
    if bmi is not None:
        st.write(f"- Estimated body mass index (BMI): **{bmi:.1f}**")
    else:
        st.write("- Estimated body mass index (BMI): **Not provided**")

    st.markdown("---")
    st.markdown("### Medical history")
    for key, question in HISTORY_QUESTIONS.items():
        value = a.get(key, False)
        st.write(f"- {question} — **{'Yes' if value else 'No'}**")

    st.markdown("---")
    st.markdown("### Symptoms")
    for key, question in SYMPTOM_QUESTIONS.items():
        value = a.get(key, False)
        st.write(f"- {question} — **{'Yes' if value else 'No'}**")

    st.markdown("---")
    st.markdown("### Lifestyle")
    for key, (question, options) in LIFESTYLE_QUESTIONS.items():
        val = a.get(key, None)
        if val is None:
            st.write(f"- {question} — **Not provided**")
        else:
            st.write(f"- {question} — **{val}**")

    st.markdown("---")
    st.markdown("### Vital signs (if provided)")
    for key, label in VITAL_FIELDS.items():
        val = a.get(key, None)
        if val is None:
            st.write(f"- {label} — **Not provided**")
        else:
            st.write(f"- {label} — **{val:.1f}**")

    st.markdown("---")
    st.markdown("### Screening-style result")

    if result is None:
        st.warning(
            "No heart sound has been analyzed yet in this session. "
            "Please go back to Step 7, upload a recording, and run the analysis."
        )
    else:
        st.write(f"- Overall label: **{result['label']}**")
        st.write(f"- Category within this prototype: **{result['severity']}**")
        st.write(f"- Combined probability of abnormal finding: `{result['p_abn']:.2f}`")
        st.write(f"- Audio-based abnormality score: `{result['p_abn_audio']:.2f}`")
        st.write(f"- History and symptom risk index: `{result['history_symptom_risk']:.2f}`")
        st.write(f"- Lifestyle risk score: `{result['lifestyle_score']:.2f}`")
        st.write(f"- Vitals risk score: `{result['vitals_score']:.2f}`")

        st.caption(
            "This summary comes from a research and education prototype. "
            "It is not a diagnosis and cannot confirm or rule out any heart condition. "
            "If you have concerns or symptoms, you should speak with a licensed healthcare professional."
        )

    st.button("⬅ Back", on_click=prev_page)


# ------------ FOOTER ------------
st.markdown(
    """
    <div class="footer">
        <span>Made by Ishita 💙</span>
    </div>
    """,
    unsafe_allow_html=True,
)
