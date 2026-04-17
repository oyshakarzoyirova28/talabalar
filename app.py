import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.title("🎓 Talaba Natijasi Bashorati")

@st.cache_data
def load_data():
    df = pd.read_csv('student_performance_updated_1000.csv')
    df.columns = [c.strip() for c in df.columns]
    return df

try:
    df = load_data()
    X = df[['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'Study Hours']]
    y = df['FinalGrade']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    att = st.sidebar.slider("Davomat (%)", 0, 100, 85)
    prev = st.sidebar.slider("Oldingi ball", 0, 100, 75)
    
    if st.button("Hisoblash"):
        pred = model.predict([[att, 20, prev, 5]]) # Namuna uchun qolgan qiymatlar 20 va 5
        st.success(f"Bashorat: {pred[0]:.2f}")
except Exception as e:
    st.error(f"Faylni yuklang: {e}")
