import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, and expected columns
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# Set page layout
st.set_page_config(page_title="Personality Predictor", layout="wide")

# Title and subtitle
st.markdown("<h1 style='color:#6C63FF'>🧠 Personality Type Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color:gray'>Discover whether you're more of an <b>Introvert</b> or <b>Extrovert</b> based on your social behavior!</h4>", unsafe_allow_html=True)
st.markdown("---")

# Helper function
def yes_no_to_int(choice):
    return 1 if choice == "Yes" else 0

# Input section
with st.expander("📋 Fill Out Your Information", expanded=True):
    col1, col2 = st.columns(2)

    with col1:
        stage_fear = st.selectbox("🎤 Do you have stage fear?", ["Yes", "No"])
        going_outside = st.number_input(
            "🚶‍♂️ How many times do you go outside in a day?", 
            min_value=0, max_value=10, value=1, step=1
        )
        friends_circle_size = st.number_input(
            "👥 Number of friends you have:", 
            min_value=0, max_value=200, value=10, step=1
        )

    with col2:
        drained_after_socializing = st.selectbox("😓 Do you feel drained after socializing?", ["Yes", "No"])
        social_events = st.number_input(
            "🎉 How many social events have you attended?", 
            min_value=0, max_value=100, value=5, step=1
        )

# Predict button
if st.button("🔍 Predict Personality"):
    # Prepare data
    data = {
        "Stage_fear": yes_no_to_int(stage_fear),
        "Social_event_attendance": social_events,
        "Going_outside": going_outside,
        "Drained_after_socializing": yes_no_to_int(drained_after_socializing),
        "Friends_circle_size": friends_circle_size
    }

    df = pd.DataFrame([data])
    df = df[expected_columns]
    scaled_data = scaler.transform(df)
    prediction = model.predict(scaled_data)

    # Result styling
    st.markdown("---")
    if prediction[0] == 1:
        st.success("🧠 You are an **INTROVERT**", icon="🧘")
        st.info("You may prefer quiet environments, deep thinking, and meaningful one-on-one conversations.")
    else:
        st.success("🎉 You are an **EXTROVERT**", icon="🥳")
        st.info("You likely enjoy social gatherings, team activities, and vibrant conversations.")

# Footer
st.markdown("---")
st.markdown("<small style='color:gray'>Created with ❤️ using Streamlit</small>", unsafe_allow_html=True)
