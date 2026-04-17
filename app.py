import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Sahifa sozlamalari
st.set_page_config(page_title="Talaba Bashorat Tizimi", layout="wide")

st.title("🎓 Talabalar natijasi: Bashorat va Grafik tahlil")
st.markdown("---")

# Ma'lumotlarni yuklash
@st.cache_data
def load_data():
    df = pd.read_csv('student_performance_updated_1000.csv')
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=['FinalGrade'])
    features = ['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'Study Hours']
    for col in features:
        df[col] = df[col].fillna(df[col].mean())
    return df, features

try:
    df, features = load_data()
    X = df[features]
    y = df['FinalGrade']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Sidebar - Ma'lumot kiritish
    st.sidebar.header("⚙️ Ko'rsatkichlarni kiring")
    att = st.sidebar.slider("Davomat (%)", 0, 100, 85)
    study_w = st.sidebar.number_input("Haftalik dars soati", 0, 100, 20)
    prev_g = st.sidebar.slider("Oldingi imtihon bali", 0, 100, 75)
    study_h = st.sidebar.number_input("Kunlik dars soati", 0, 24, 5)

    if st.button("🚀 Hisoblash va Tahlil qilish"):
        input_vals = [att, study_w, prev_g, study_h]
        prediction = model.predict([input_vals])[0]
        
        # 1-QADAM: BASHORAT NATIJASI
        st.subheader("1. Bashorat qilingan natija")
        st.metric(label="Yakuniy ball (Taxminan)", value=f"{prediction:.2f}")
        st.markdown("---")

        # 2-QADAM: DIAGRAMMA (Bar Chart)
        st.subheader("2. Faktorlar diagrammasi")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        sns.barplot(x=['Davomat', 'Haftalik dars', 'Oldingi ball', 'Kunlik dars'], 
                    y=input_vals, palette='magma', ax=ax1)
        ax1.set_title("Kiritilgan faktorlar nisbati")
        st.pyplot(fig1)
        st.markdown("---")

        # 3-QADAM: BOG'LIQLIK CHIZMASI (Regression Line)
        # Davomat va Yakuniy ball o'rtasidagi bog'liqlikni trend chizig'i bilan ko'rsatamiz
        st.subheader("3. Davomat va Natija bog'liqlik chizmasi")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.regplot(data=df.sample(200), x='AttendanceRate', y='FinalGrade', 
                    scatter_kws={'alpha':0.3}, line_kws={'color':'red'}, ax=ax2)
        
        # Foydalanuvchi nuqtasini chizmada ko'rsatish
        ax2.scatter(att, prediction, color='yellow', s=200, label='Sizning holatingiz', edgecolors='black')
        ax2.legend()
        ax2.set_title("Davomatning yakuniy ballga ta'siri (Trend)")
        st.pyplot(fig2)

except Exception as e:
    st.error(f"Xatolik: {e}. requirements.txt faylida seaborn va matplotlib yozilganini tekshiring.")
