import streamlit as st
import requests

# -------- CONFIG --------
API_URL = "https://credit-card-complaint-classifier-1.onrender.com/predict" 

st.set_page_config(page_title="Credit Card Complaint Classifier", page_icon="💳", layout="centered")

st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
        padding: 2rem 4rem;
        max-width: 700px;
        margin: auto;
        border-radius: 10px;
        box-shadow: 0 8px 20px rgb(0 0 0 / 0.1);
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #004999;
    }
    </style>
""", unsafe_allow_html=True)

with st.container():
    st.title("💳 Credit Card Complaint Classifier")
    st.write("Enter your complaint below, and get instant classification into categories like billing, fraud, customer service, and more.")

    complaint_text = st.text_area("Enter complaint text here...", height=150, max_chars=500)

    if st.button("Classify Complaint"):
        if not complaint_text.strip():
            st.warning("Please enter a complaint text to classify.")
        else:
            with st.spinner("Classifying..."):
                try:
                    response = requests.post(API_URL, json={"text": complaint_text})
                    if response.status_code == 200:
                        result = response.json()
                        prediction = result.get("prediction", "Unknown")
                        confidence = result.get("confidence", None)
                        
                        if confidence is not None:
                            st.success(f"Prediction: **{prediction.capitalize()}**\n\nConfidence Score: **{confidence * 100:.2f}%**")
                        else:
                            st.success(f"Prediction: **{prediction.capitalize()}**")

                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to API: {e}")

    st.markdown("---")
    st.write("**Note:** This app uses a FastAPI backend for model inference.")

    st.caption("Built  by Tanish Sharma | Model powered by Hugging Face & Transformers")
