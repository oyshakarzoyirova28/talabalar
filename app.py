import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Sahifa dizayni
st.set_page_config(page_title="Talaba Bashorat Tizimi", layout="wide")

st.title("🎓 Talabalar natijasi va vizual tahlili")
st.markdown("---")

# Ma'lumotlarni yuklash va tayyorlash
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
    
    # Modelni o'qitish
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Sidebar - Ma'lumot kiritish
    st.sidebar.header("Kiritish paneli")
    att = st.sidebar.slider("Davomat (%)", 0, 100, 85)
    study_w = st.sidebar.number_input("Haftalik dars (soat)", 0, 100, 20)
    prev_g = st.sidebar.slider("Oldingi ball", 0, 100, 75)
    study_h = st.sidebar.number_input("Kunlik dars (soat)", 0, 24, 5)

    # ASOSIY QISM
    if st.button("🚀 Hisoblash va Diagrammani ko'rish"):
        input_vals = [att, study_w, prev_g, study_h]
        prediction = model.predict([input_vals])[0]
        
        # 1. Natijani ko'rsatish
        st.success(f"### Bashorat qilingan yakuniy ball: {prediction:.2f}")

        # 2. DIAGRAMMA (Chizma) QISMI
        st.markdown("#### 📊 Faktorlar tahlili diagrammasi")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = sns.color_palette('viridis', len(features))
        
        # Ustunli diagramma chizish
        sns.barplot(x=['Davomat', 'Haftalik o\'qish', 'Oldingi ball', 'Kunlik dars'], 
                    y=input_vals, palette=colors, ax=ax)
        
        # Diagramma ustiga qiymatlarni yozish
        for i, v in enumerate(input_vals):
            ax.text(i, v + 1, str(v), ha='center', fontweight='bold')
            
        ax.set_ylim(0, 110)
        ax.set_ylabel("Qiymatlar")
        ax.set_title("Siz kiritgan ko'rsatkichlar nisbati")
        
        # Chizmani Streamlit-ga chiqarish
        st.pyplot(fig)
        
        # 3. Qo'shimcha tahlil
        st.info(f"Ushbu talaba uchun model {prediction:.2f} ballni bashorat qildi. "
                f"Bu ko'rsatkich asosan {att}% davomat va {prev_g} ballik bazaga asoslangan.")

except Exception as e:
    st.error(f"Xatolik yuz berdi: {e}. Iltimos, requirements.txt ichida matplotlib borligini tekshiring.")
