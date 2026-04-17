import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# SAHIFA SOZLAMALARI VA DIZAYNI
# ==========================================
st.set_page_config(
    page_title="Talaba Natijasi Bashorati | ML",
    page_icon="🎓",
    layout="wide", # Kengaytirilgan rejim
    initial_sidebar_state="expanded"
)

# Professional vizual uslub (Matplotlib uchun)
plt.style.use('seaborn-v0_8-pastel')

# Sahifa sarlavhasi
col1, col2 = st.columns([1, 15])
with col1:
    st.markdown("# 🎓")
with col2:
    st.title("Talabalar akademik natijasini bashorat qilish tizimi")
st.markdown("---")

# ==========================================
# DATASETNI YUKLASH VA TOZALASH (Preprocessing)
# ==========================================
@st.cache_data
def load_and_preprocess_data():
    try:
        # Fayl nomini tekshiring
        df = pd.read_csv('student_performance_updated_1000.csv')
        
        # 1. Ustun nomlaridagi ortiqcha bo'shliqlarni olib tashlaymiz
        df.columns = [c.strip() for c in df.columns]
        
        # 2. MUHIM: Maqsadli ustun (FinalGrade) dagi NaN qiymatlarni o'chiramiz
        df = df.dropna(subset=['FinalGrade'])
        
        # 3. Faktor ustunlarini tanlaymiz
        features = ['AttendanceRate', 'StudyHoursPerWeek', 'PreviousGrade', 'Study Hours']
        
        # 4. Faktor ustunlaridagi NaN qiymatlarni o'rtacha raqam bilan to'ldiramiz
        for col in features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
            else:
                st.error(f"❌ Xatolik: Datasetda '{col}' ustuni topilmadi.")
                st.stop()
                
        # 5. Model uchun ma'lumotlarni tayyorlaymiz
        X = df[features]
        y = df['FinalGrade']
        
        # Modelni o'qitish
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Model aniqligini hisoblash (Kurs ishi uchun)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        return df, model, features, r2, mae
        
    except FileNotFoundError:
        st.error("❌ Xatolik: `student_performance_updated_1000.csv` fayli topilmadi.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Kutilmagan xatolik yuz berdi: {e}")
        st.stop()

# Ma'lumotlarni tayyorlaymiz
df, model, features, r2_score_val, mae_score = load_and_preprocess_data()

# ==========================================
# FOYDALANUVCHI INTERFEYSI (Sidebar)
# ==========================================
st.sidebar.markdown("# 🔧 Kiritish paneli")
st.sidebar.markdown("Mashinali o'rganish modeli uchun ma'lumotlarni kiring:")
st.sidebar.markdown("---")

# Slider va Number input yordamida faktorlarni kiritish
attendance = st.sidebar.slider("📉 Davomat darajasi (%)", 0.0, 100.0, 85.0)
prev_grade = st.sidebar.slider("📊 Oldingi imtihon bali", 0.0, 100.0, 75.0)
study_week = st.sidebar.number_input("📚 Haftalik dars vaqti (soat)", 0, 100, 20)
study_hours = st.sidebar.number_input("⏳ Kunlik qo'shimcha dars (soat)", 0, 24, 5)

# O'zbekcha nomlar grafika uchun
feature_names_uz = ['Davomat', 'Haftalik o\'qish', 'Oldingi ball', 'Kunlik dars']

st.sidebar.markdown("---")
# Model aniqligini Sidebar-da ko'rsatish (Professional ko'rinish)
with st.sidebar.expander("✅ Model samaradorligi (Kurs ishi uchun)"):
    st.write(f"Model: **Random Forest**")
    st.write(f"Aniqlik (R²): **{r2_score_val:.2f}**")
    st.write(f"Xatolik (MAE): **{mae_score:.2f}**")

# ==========================================
# ASOSIY QISM - BASHORAT VA TAHLIL
# ==========================================
st.markdown("### 🔍 Bashorat va Tahlil Natijalari")
st.markdown("Pastdagi tugmani bosing, tizim soniyalar ichida sizning natijangizni hisoblab chiqadi.")

if st.button("🚀 Natijani bashorat qilish", type="primary"):
    
    # 1. Bashorat qilish
    input_data = np.array([[attendance, study_week, prev_grade, study_hours]])
    prediction = model.predict(input_data)[0]
    
    # 2. Bashorat natijasini chiroyli chiqarish (Metrics)
    st.markdown("---")
    main_col1, main_col2, main_col3 = st.columns([1, 1, 1])
    
    with main_col1:
        st.metric(label="Bashorat qilingan yakuniy ball", value=f"{prediction:.2f}")
    
    with main_col2:
        # Ballga qarab holatni aniqlash
        if prediction >= 85:
            grade_status = "Namunali (A)"
            st.markdown(f"#### Holat: <span style='color:green;'>{grade_status}</span>", unsafe_allow_html=True)
        elif prediction >= 71:
            grade_status = "Yaxshi (B)"
            st.markdown(f"#### Holat: <span style='color:blue;'>{grade_status}</span>", unsafe_allow_html=True)
        elif prediction >= 60:
            grade_status = "Qoniqarli (C)"
            st.markdown(f"#### Holat: <span style='color:orange;'>{grade_status}</span>", unsafe_allow_html=True)
        else:
            grade_status = "Yomon (D/F)"
            st.markdown(f"#### Holat: <span style='color:red;'>{grade_status}</span>", unsafe_allow_html=True)
    
    with main_col3:
        if prediction >= 85:
            st.balloons()
            st.write("Ajoyib! Siz namunali talabasiz.")
        elif prediction >= 60:
            st.write("Natijangiz yaxshi. Shunday davom eting.")
        else:
            st.write("Diqqat! Natijangiz past. Darslarga ko'proq e'tibor bering.")

    # 3. Bashorat tahlili diagrammasi (Vizualizatsiya)
    st.markdown("---")
    st.markdown("#### 📊 Bashorat tahlili: Faktorlarning kiritilgan qiymatlari")
    
    # Ma'lumotlarni grafika uchun tayyorlash
    chart_data = pd.DataFrame({
        'Faktor': feature_names_uz,
        'Qiymat': [attendance, study_week, prev_grade, study_hours]
    })
    
    # Matplotlib yordamida chiroyli grafika chizish
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Bar chart chizish
    bars = ax.bar(chart_data['Faktor'], chart_data['Qiymat'], color=['#3498db', '#2ecc71', '#f1c40f', '#e74c3c'])
    
    # Barlar ustiga raqamlarni yozish
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Grafik dizayni
    ax.set_title("Kiritilgan faktorlar qiymatlari solishtiruvi", fontsize=14, fontweight='bold')
    ax.set_ylabel("Qiymat / Foiz", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=10)
    
    # GrafikaStreamlit-da ko'rsatish
    st.pyplot(fig)

    # 4. Talaba uchun tavsiyalar (Yangi funksiya)
    st.markdown("---")
    st.markdown("#### 💡 Talaba uchun akademik tavsiyalar")
    
    with st.expander("Batafsil tavsiyalar:", expanded=True):
        if attendance < 70:
            st.warning("🆘 Davomatingiz juda past. Darslarga qatnashish yakuniy ballga eng ko'p ta'sir qiluvchi faktorlardan biridir.")
        
        if prev_grade < 60:
            st.warning("⚠️ Oldingi imtihon ballaringiz past. Mavzularni qayta ko'rib chiqishingiz va tushunarsiz joylarni o'qituvchilardan so'rashingiz tavsiya etiladi.")
            
        if study_week < 15:
            st.info("ℹ️ Haftalik o'qish vaqti kam. O'zlashtirishni yaxshilash uchun kunlik dars soatini oshirishingiz kerak.")
        
        if prediction >= 85:
            st.success("✅ Siz namunali natija ko'rsatyapsiz. Kelajakda stipendiya, xalqaro dasturlar va grantlarga hujjat topshirishingiz mumkin.")
        elif prediction >= 71:
            st.success("✅ Natijangiz yaxshi. Biroz ko'proq harakat qilsangiz, namunali natijaga erishishingiz mumkin.")

else:
    # Tugma bosilishidan oldin ko'rinadigan qism
    st.markdown("---")
    st.info("Tahlil natijalari bu yerda paydo bo'ladi. Mashinali o'rganish modeli siz kiritgan ma'lumotlarni orqa fonda hisob-kitob qiladi.")
    
    # Ilova haqida tushuntirish
    st.markdown("""
    #### Ilova qanday ishlaydi?
    1.  Chap tarafdagi paneldan **davomatingiz, haftalik o'qish vaqti, oldingi ballaringiz va kunlik dars soati**ni kiritasiz.
    2.  Model tarixiy ma'lumotlar (dataset) asosida kiritilgan ma'lumotlarni tahlil qiladi.
    3.  Tizim Random Forest algoritmi yordamida yakuniy balingizni **bashorat** qiladi va vizual tahlilni ko'rsatadi.
    """)

# ==========================================
# SAHIFA PASTKI QISMI (Footer)
# ==========================================
st.markdown("---")
st.markdown("<div style='text-align: center;'>Talabalar natijasini bashorat qilish tizimi © 2024 | ML Loyihasi</div>", unsafe_allow_html=True)
