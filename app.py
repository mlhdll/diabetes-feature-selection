import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_manager import load_data, preprocess_data
from feature_selector import select_features
from model_trainer import train_evaluate

# Sayfa yapılandırması
st.set_page_config(page_title="Diyabet Özellik Seçimi", layout="wide")

# Başlık ve Açıklama
st.title("🔍 Diyabet Teşhisi: Özellik Seçimi ve Tahmin")
st.markdown("""
Bu uygulama, diyabet teşhisinde hangi klinik parametrelerin en belirleyici olduğunu çeşitli özellik seçimi yöntemleri kullanarak araştırır.
Ayrıca kendi riskinizi tahmin etmenizi sağlar.
""")

# Veri Yükleme
@st.cache_data
def get_data():
    df, error = load_data()
    if error:
        st.error(error)
        return None, None, None
    X, y = preprocess_data(df)
    return df, X, y

df, X, y = get_data()

if df is not None:
    # Kenar Çubuğu Kontrolleri
    st.sidebar.header("⚙️ Ayarlar")
    
    # Özellik Seçimi Yöntemi
    fs_method = st.sidebar.selectbox(
        "Özellik Seçimi Yöntemini Seçin",
        ['RFE (Recursive Feature Elimination)', 'Mutual Information', 'LASSO (L1 Regularization)']
    )
    
    # Seçilecek Özellik Sayısı
    k_features = st.sidebar.slider("Seçilecek Özellik Sayısı (k)", min_value=1, max_value=len(X.columns), value=5)
    
    # Model Türü
    model_type = st.sidebar.selectbox(
        "Değerlendirme İçin Model Seçin",
        ['Logistic Regression', 'Random Forest']
    )

    # --- Ana İçerik ---

    # 1. Özellik Seçimi Sonuçları
    st.header(f"1. Özellik Seçimi: {fs_method}")
    
    selected_features, scores = select_features(fs_method, X, y, k=k_features)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Seçilen Özellikler")
        for i, f in enumerate(selected_features, 1):
            st.write(f"{i}. **{f}**")
            
    with col2:
        st.subheader("Özellik Önemi / Puanları")
        if scores:
            # Görselleştirme için puanları sırala
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            score_df = pd.DataFrame(sorted_scores, columns=['Feature', 'Score'])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Score', y='Feature', data=score_df, palette='viridis', ax=ax)
            ax.set_title(f"{fs_method} Kullanılarak Özellik Önemi")
            ax.set_xlabel("Göreceli Önem (%)")
            st.pyplot(fig)

    # 2. Model Performansı
    st.header(f"2. Model Performansı: {model_type}")
    st.markdown(f"Model **sadece seçilen {k_features} özellik** kullanılarak değerlendiriliyor.")
    
    # Modeli seçilen özelliklerle eğit
    X_selected = X[selected_features]
    results = train_evaluate(model_type, X_selected, y)
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("Doğruluk (Accuracy)", f"{results['accuracy']:.2%}")
    m_col2.metric("F1 Skoru", f"{results['f1_score']:.2%}")
    
    with m_col3:
        st.write("**Karmaşıklık Matrisi (Confusion Matrix)**")
        cm = results['confusion_matrix']
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel('Tahmin Edilen')
        ax_cm.set_ylabel('Gerçek')
        st.pyplot(fig_cm)

    # 3. Canlı Tahmin
    st.header("3. Diyabet Riskinizi Tahmin Edin")
    st.markdown("Tahmin almak için klinik değerlerinizi aşağıya girin.")
    
    # Giriş formu
    with st.form("prediction_form"):
        # Girişler için sütunlar oluştur
        i_col1, i_col2, i_col3, i_col4 = st.columns(4)
        
        user_input = {}
        
        # Veri seti min/max değerlerine veya standart tıbbi aralıklara göre aralıkları tanımla
        with i_col1:
            user_input['Pregnancies'] = st.number_input("Gebelik Sayısı (Pregnancies)", min_value=0, max_value=20, value=1)
            user_input['Glucose'] = st.number_input("Glikoz (Glucose) mg/dL", min_value=0, max_value=300, value=120)
        with i_col2:
            user_input['BloodPressure'] = st.number_input("Kan Basıncı (BloodPressure) mm Hg", min_value=0, max_value=200, value=70)
            user_input['SkinThickness'] = st.number_input("Cilt Kalınlığı (SkinThickness) mm", min_value=0, max_value=100, value=20)
        with i_col3:
            user_input['Insulin'] = st.number_input("İnsülin (Insulin) mu U/ml", min_value=0, max_value=900, value=80)
            user_input['BMI'] = st.number_input("Vücut Kitle İndeksi (BMI)", min_value=0.0, max_value=70.0, value=25.0)
        with i_col4:
            user_input['DiabetesPedigreeFunction'] = st.number_input("Diyabet Soyağacı Fonksiyonu", min_value=0.0, max_value=3.0, value=0.5)
            user_input['Age'] = st.number_input("Yaş (Age)", min_value=0, max_value=120, value=30)
            
        submit_button = st.form_submit_button("Riski Tahmin Et")
        
        if submit_button:
            # Giriş için veri çerçevesi oluştur
            input_df = pd.DataFrame([user_input])
            
            # Bu girişi eğitim verisiyle AYNI ölçekleyiciyi kullanarak ölçeklendirmemiz gerekiyor.
            # Gerçek bir uygulamada ölçekleyiciyi kaydederdik. Bu demo için orijinal veriye uydurup dönüştüreceğiz.
            
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            # Ölçekleyiciyi uydurmak için orijinal ölçeklenmemiş X'e ihtiyacımız var
            X_orig = df.drop(columns=['Outcome'])
            scaler.fit(X_orig)
            
            input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
            
            # Sadece model tarafından kullanılan özellikleri seç
            input_selected = input_scaled[selected_features]
            
            # Tahmin et
            prediction = results['model'].predict(input_selected)[0]
            prob = results['model'].predict_proba(input_selected)[0][1]
            
            st.subheader("Tahmin Sonucu")
            if prediction == 1:
                st.error(f"Yüksek Diyabet Riski (Olasılık: {prob:.2%})")
                st.markdown("Lütfen bir sağlık uzmanına danışın.")
            else:
                st.success(f"Düşük Diyabet Riski (Olasılık: {prob:.2%})")
                st.markdown("Sağlıklı yaşam tarzına devam edin!")

else:
    st.warning("Veri yüklenemedi.")
