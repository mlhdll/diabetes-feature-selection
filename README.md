# Diyabet Teşhisi İçin Özellik Seçimi (Feature Selection for Diabetes Diagnosis)

Bu proje, Pima Indians Diabetes Veri Seti kullanılarak diyabet teşhisinde en etkili klinik özellikleri belirlemeyi ve makine öğrenmesi modelleriyle risk tahmini yapmayı amaçlayan bir Streamlit uygulamasıdır.

## 🚀 Özellikler

*   **Özellik Seçimi:** 3 farklı yöntem ile en önemli özellikleri belirleme:
    *   RFE (Recursive Feature Elimination)
    *   Mutual Information (Karşılıklı Bilgi)
    *   LASSO (L1 Regularization)
*   **Model Karşılaştırma:** Seçilen özelliklerle modellerin performansını test etme:
    *   Lojistik Regresyon (Logistic Regression)
    *   Rastgele Orman (Random Forest)
*   **Görselleştirme:** Özelliklerin önem derecelerini gösteren interaktif grafikler.
*   **Canlı Tahmin:** Kullanıcıdan alınan verilerle diyabet riski tahmini.

## 📂 Proje Yapısı

*   `app.py`: Ana Streamlit uygulaması.
*   `data_manager.py`: Veri yükleme ve ön işleme işlemleri.
*   `feature_selector.py`: Özellik seçimi algoritmaları.
*   `model_trainer.py`: Model eğitimi ve değerlendirmesi.
*   `diabetes.csv`: Kullanılan veri seti (`data/` klasörü içinde).

## 🛠️ Kurulum

1.  Bu depoyu (repository) klonlayın:
    ```bash
    git clone https://github.com/KULLANICI_ADINIZ/diabetes-feature-selection.git
    cd diabetes-feature-selection
    ```

2.  Gerekli kütüphaneleri yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

3.  Uygulamayı çalıştırın:
    ```bash
    streamlit run app.py
    ```

## 📊 Kullanılan Teknolojiler

*   Python
*   Streamlit
*   Scikit-learn
*   Pandas & NumPy
*   Matplotlib & Seaborn


