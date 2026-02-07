import streamlit as st
import joblib

st.set_page_config(page_title="Fake News Detector", layout="wide")

# Title
st.title("📰 Fake News Detection App")

# Load model + vectorizer
model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Initialize session state variable
if "news_text" not in st.session_state:
    st.session_state["news_text"] = ""

# Callback function to clear text
def clear_text():
    st.session_state["news_text"] = ""

# Input text tied to session_state
news_text = st.text_area("✍️ Enter news text here:", key="news_text", height=180)

# Buttons
col1, col2 = st.columns(2)
with col1:
    predict_btn = st.button("🧠 Predict")
with col2:
    st.button("🧹 Clear text", on_click=clear_text)

# Prediction logic
if predict_btn:
    if not st.session_state["news_text"].strip():
        st.warning("⚠️ Please enter some text first!")
    else:
        with st.spinner("🔄 Predicting..."):
            vect = tfidf.transform([st.session_state["news_text"]])
            pred = model.predict(vect)[0]

        if pred == "FAKE":
            st.error("🚫 This news looks *FAKE*.")
        else:
            st.success("✅ This news looks *REAL*.")