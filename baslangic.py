### KUTUPHANELER (START) ###
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
import seaborn as sns
### KUTUPAHANELER (END) ###

### BASLANGIC (START) ###
def app():

    # Sekmeler (tabs) ile dört ana bölümü bir arada sunuyoruz
    tabs = st.tabs([
        "Hakkımda",
        "Dinamik Veri Analizi",
        "Veri Analizi & Ön İşleme",
        "Model Eğitimi & Değerlendirme"
    ])

    # 1) Hakkımda bölümü
    with tabs[0]:
        card_col, action_col = st.columns([3, 1], gap="medium")
        with card_col:
            st.markdown(
                """
                <div style="
                    background-color: rgb(0,104,201);
                    padding: 2rem;
                    border-radius: 12px;
                    color: white;
                ">
                    <h2 style="margin: 0 0 0.5rem; font-size:1.75rem;">Baran Kanat</h2>
                    <ul style="list-style:none; padding:0; margin:0; line-height:1.6;">
                        <li><strong>Üniversite:</strong> Necmettin Erbakan Üniversitesi</li>
                        <li><strong>Bölüm:</strong> Bilgisayar Mühendisliği</li>
                        <li><strong>Ortalama:</strong> 3.70</li>
                        <li><strong>Durum:</strong> 4. Sınıf</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
        with action_col:
            st.markdown(
                """
                <div style="display:flex; flex-direction:column; gap:1rem;">
                    <a href="https://www.linkedin.com/in/baran-kanat/" target="_blank"
                    style="
                        background: rgb(40,103,178);
                        color:white;
                        padding:0.6rem 1rem;
                        border-radius:8px;
                        text-decoration:none;
                        text-align:center;
                        font-weight:600;
                        display:block;
                    ">
                    🔗 LinkedIn
                    </a>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.write("")
            with open("Baran_Kanat_CV.pdf", "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="📄 CV İndir",
                data=pdf_bytes,
                file_name="Baran_Kanat_CV.pdf",
                mime="application/pdf",
                use_container_width=True
            )

    # 2) Dinamik Veri Analizi bölümü
    with tabs[1]:
        st.header("Dinamik Veri Analizi")
        st.write(
            """
            Breast Cancer Wisconsin (Diagnostic) Data Set üzerinde dinamik bir arayüz oluşturdum:
            
            - Veri setini istediğiniz gibi düzenleyebileceğiniz bir yapı  
            - Temel istatistikleri, grafikleri ve filtreleri anlık olarak uygulama  

            Böylece aynı veri setiyle farklı ön işleme ve analiz tekniklerini rahatça test edebilirsiniz.
            """
        )

    # 3) Veri Analizi & Ön İşleme bölümü
    with tabs[2]:
        st.header("Veri Analizi & Ön İşleme")
        st.write(
            """
            Breast Cancer Wisconsin (Diagnostic) Data Set için yaptığım veri ön işleme adımları:

            - Eksik değer kontrolü ve uygun imputasyon  
            - Aykırı değer analizi ve temizleme  
            - Korelasyon analizi ile yüksek ilişkili özellik seçimi  
            - Özellik ölçekleme (StandardScaler, MinMaxScaler)  
            - Kategorik etiketleri numeric’e dönüştürme (B=1, M=0)  
            - One-hot encoding, label encoding vs.  
            """
        )

    # 4) Model Eğitimi & Değerlendirme bölümü
    with tabs[3]:
        st.header("Model Eğitimi & Değerlendirme")
        st.write(
            """
            Ön işleme aşamasından elde ettiğim veri seti üzerinde:

            - Geleneksel makine öğrenme modelleri  
              (KNN, SVM, Random Forest, Logistic Regression, XGBoost)  
            - Derin öğrenme modelleri  
              (CNN, LSTM, MLP, TABNET)  

            Eğitim, test ve 10-katlı cross-validation ile karşılaştırmalı değerlendirme yaptım.  
            Doğruluk, precision, recall, f1-score ölçütleriyle performansları inceledim.
            """
        )
        # Tüm kodların bulunduğu klasör bilgisi
        st.info(
            "📂 Tüm kodlar `Model-VeriAnalizi` klasöründe `.ipynb` dosyası olarak yer almaktadır; tekrar çalıştırmaya gerek kalmaz."
        )

### BASLANGIC (END) ###