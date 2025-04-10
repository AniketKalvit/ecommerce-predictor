import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🛒 E-Commerce Purchase Predictor")

# User Inputs
time_spent = st.slider("Time Spent on Site (min)", 0, 100, 10)
pages_visited = st.slider("Pages Visited", 1, 10, 2)
country = st.selectbox("Country", ['US', 'IN', 'UK'])
is_returning = st.radio("Is Returning User?", ['Yes', 'No'])

# Encode country manually
country_IN = int(country == 'IN')
country_UK = int(country == 'UK')

# Encode returning user
is_returning_user = 1 if is_returning == 'Yes' else 0

# Feature engineering
engagement_score = time_spent * pages_visited

# Final features
X = np.array([[time_spent, pages_visited, is_returning_user,
               country_IN, country_UK, engagement_score]])

# Scale features
X_scaled = scaler.transform(X)

# Predict
prediction = model.predict(X_scaled)[0]
prob = model.predict_proba(X_scaled)[0][1]

st.markdown("---")
if st.button("Predict"):
    if prediction == 1:
        st.success(f"🎯 This user is likely to make a purchase! (Confidence: {prob:.2f})")
    else:
        st.warning(f"🚫 This user is unlikely to make a purchase. (Confidence: {prob:.2f})")