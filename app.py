import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Sahifa sarlavhasi va dizayni
st.set_page_config(page_title="Talaba Bashorat Tizimi", page_icon="🎓")

st.title("🎓 Talabalar natijasini bashorat qilish tizimi")
st.markdown("---")

# Ma'lumotlarni yuklash va TOZALASH
@st.cache_data
def load_and_clean_data():
    # Fayl nomini tekshiring
    df = pd.read_csv('student_performance_updated_1000.csv')
    
    # 1. Ustun nomlaridagi ortiqcha bo'shliqlarni olib tashlaymiz
    df.columns = [c.strip() for c in df.columns]
    
    # 2. Maqsadli ustun (FinalGrade) dagi NaN qiymatli qatorlarni butunlay o'chiramiz
    # Chunki javobi yo'q qator bilan modelni o'qitib bo'lmaydi
    df = df.dropna(subset=['FinalGrade'])
    
    # 3. Faktor ustunlaridagi NaN qiymatlarni o'rtacha raqam bilan to'ldiramiz
    features = ['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'Study Hours']
    for col in features:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
            
    return df, features

try:
    # Ma'lumotlarni tayyorlaymiz
    df, features = load_and_clean_data()
    
    X = df[features]
    y = df['FinalGrade']

    # Modelni o'qitish (Random Forest)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Foydalanuvchi interfeysi (Sidebar)
    st.sidebar.header("Talaba ma'lumotlarini kiriting:")
    
    att = st.sidebar.slider("Davomat darajasi (%)", 0.0, 100.0, 85.0)
    study_w = st.sidebar.number_input("Haftalik dars vaqti (soat)", 0, 100, 20)
    prev_g = st.sidebar.slider("Oldingi imtihon bali", 0.0, 100.0, 75.0)
    study_h = st.sidebar.number_input("Kunlik qo'shimcha dars (soat)", 0, 24, 5)

    # Bashorat tugmasi
    if st.button("Natijani bashorat qilish"):
        input_data = np.array([[att, study_w, prev_g, study_h]])
        prediction = model.predict(input_data)
        
        # Natijani chiqarish
        st.success(f"### Bashorat qilingan yakuniy ball: {prediction[0]:.2f}")
        
        # Grafik ko'rinishida ko'rsatish
        st.info("Kiritilgan ma'lumotlar tahlili:")
        chart_data = pd.DataFrame({
            'Faktorlar': features,
            'Qiymatlar': [att, study_w, prev_g, study_h]
        })
        st.bar_chart(chart_data.set_index('Faktorlar'))

except FileNotFoundError:
    st.error("❌ Xatolik: `student_performance_updated_1000.csv` fayli topilmadi. GitHub-ga yuklaganingizni tekshiring.")
except Exception as e:
    st.error(f"❌ Kutilmagan xatolik yuz berdi: {e}")
