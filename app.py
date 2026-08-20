import os
import joblib
import streamlit as st

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

APP_TITLE = "Sentiment Analysis"
APP_VERSION = "1.0.0"
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    }
    .block-container {
        max-width: 850px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }
    .hero {
        padding: 2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #312e81, #7c3aed);
        color: white;
        text-align: center;
        box-shadow: 0 12px 35px rgba(49, 46, 129, 0.18);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.4rem;
    }
    .hero p {
        margin: .65rem 0 0;
        opacity: .92;
    }
    .result-positive {
        padding: 1.2rem;
        border-radius: 16px;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        text-align: center;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .result-negative {
        padding: 1.2rem;
        border-radius: 16px;
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
        text-align: center;
        font-size: 1.15rem;
        font-weight: 700;
    }
    .footer {
        text-align: center;
        color: #64748b;
        font-size: .85rem;
        margin-top: 2rem;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing {MODEL_PATH}")
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"Missing {VECTORIZER_PATH}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    if not hasattr(model, "predict"):
        raise TypeError("model.pkl is not a valid prediction model.")
    if not hasattr(vectorizer, "transform"):
        raise TypeError(
            "vectorizer.pkl is not a valid text vectorizer. "
            "Please export the fitted vectorizer used during training."
        )

    return model, vectorizer


def predict_sentiment(text, model, vectorizer):
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))

    return str(prediction), confidence


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>💬 Sentiment Analysis</h1>
        <p>Analyze text and classify it as Positive or Negative using machine learning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load files
# ---------------------------------------------------------
try:
    model, vectorizer = load_artifacts()
    artifacts_ready = True
except Exception as exc:
    artifacts_ready = False
    st.error("The application could not load the required ML files.")
    st.info(
        "Place `model.pkl` and a fitted `vectorizer.pkl` in the same folder as `app.py`."
    )
    st.caption(f"Technical details: {exc}")

# ---------------------------------------------------------
# User interface
# ---------------------------------------------------------
st.subheader("Analyze your text")
st.write("Enter a review, comment, message, or sentence below.")

text = st.text_area(
    "Text",
    height=170,
    placeholder="Example: I really enjoyed this product. It works perfectly!",
)

col1, col2 = st.columns([3, 1])

with col1:
    analyze = st.button(
        "🔍 Analyze Sentiment",
        type="primary",
        disabled=not artifacts_ready,
    )

with col2:
    st.metric("Characters", len(text))

if analyze:
    cleaned_text = text.strip()

    if not cleaned_text:
        st.warning("Please enter some text before analyzing.")
    else:
        try:
            prediction, confidence = predict_sentiment(
                cleaned_text, model, vectorizer
            )

            normalized = prediction.lower()

            st.divider()
            st.subheader("Prediction Result")

            if "positive" in normalized:
                st.markdown(
                    '<div class="result-positive">😊 Positive Sentiment</div>',
                    unsafe_allow_html=True,
                )
            elif "negative" in normalized:
                st.markdown(
                    '<div class="result-negative">😟 Negative Sentiment</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.success(f"Prediction: {prediction}")

            if confidence is not None:
                st.progress(confidence)
                st.caption(f"Model confidence: {confidence:.1%}")

        except Exception as exc:
            st.error("Prediction failed.")
            st.caption(f"Technical details: {exc}")

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.title("About")
    st.write(
        "This Streamlit application uses a trained machine-learning model "
        "and text vectorizer to predict sentiment."
    )
    st.caption(f"Version {APP_VERSION}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown(
    '<div class="footer">Built with Python, Streamlit & scikit-learn</div>',
    unsafe_allow_html=True,
)
