### KUTUPHANELER (START) ###
import streamlit as st

import pandas as pd  # Veri analizi ve DataFrame islemleri icin kullanilir
import numpy as np  # Sayisal islemler ve array yapilari icin kullanilir
import matplotlib.pyplot as plt  # Grafik cizimi ve gorsellestirme icin kullanilir
import seaborn as sns  # Gelismis istatistiksel grafikler icin kullanilir

from sklearn.preprocessing import StandardScaler # veriyi standartlastirmak icin kullanilir

# Model Egitimi Icin #
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score 
# train_test_split: veriyi egitim ve test olarak bolmek icin kullanilir  
# GridSearchCV: en iyi model parametrelerini bulmak icin grid arama yapar (cross-validation icerir)
# cross_val_score: Modeller uzerinde cross-validation uygulamamizi saglar

from sklearn.preprocessing import LabelEncoder, StandardScaler  
# LabelEncoder: kategorik etiketleri sayisal degerlere donusturmek icin kullanilir  
# StandardScaler: veriyi ortalama=0, std=1 olacak sekilde standartlastirir (normalize eder)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay  
# accuracy_score: dogruluk oranini hesaplar  
# confusion_matrix: gercek vs tahmin siniflarin karsilastirmasini tablo olarak verir  
# classification_report: precision, recall, f1-score gibi siniflandirma metriklerini verir
# ConfusionMatrixDisplay: Grafiksel karmasiklik Matriksi

import tensorflow as tf  # TensorFlow kütüphanesi

import altair as alt # Bar grafigi icin kullanilir
### KUTUPHANELER (END) ###

### FONKSIYONLAR (START) ###
## tasarim foknsiyonlari ##
#Bilgi kutucugu
def boxInfo(content: str, border_color: str = "#ADD8E6"):
    st.markdown(f"""
    <div style='background-color: #f9f9f9; padding: 15px 15px 5px 15px; border-left: 5px solid {border_color}; border-radius: 5px'>
        {content}
    </div>
    """, unsafe_allow_html=True)

#Acikleme Kutucu
def boxExp(content: str, border_color: str = "#ADD8E6"):
    st.markdown(f"""
    <div style='background-color: #f9f9f9; padding: 15px 15px 5px 15px; border-top: 5px solid {border_color}; border-radius: 5px'>
        {content}
    </div>
    """, unsafe_allow_html=True)

def boxModelInfo(content: str, border_color: str = "#ADD8E6"):
    st.markdown(f"""
    <div style='background-color: #f9f9f9; padding-left: 5px; border-left: 5px solid {border_color}; border-radius: 5px'>
        {content}
    </div>
    """, unsafe_allow_html=True)
## ./tasarim foknsiyonlari ##

### FONKSIYONLAR (END) ###

def app():
    st.title("Geleneksel Makine Öğrenme Modelleri")
    boxExp("""
    Geleneksel makine öğrenmesi modelleri, genellikle yapılandırılmış ve önceden işlenmiş veri kümeleri üzerinde çalışır. Bu modeller, **elle seçilmiş özellikler** üzerinden istatistiksel ve karar tabanlı yaklaşımlar kullanarak öğrenir ve tahmin yapar. Yaygın özellikleri şunlardır:

    - **Şeffaflık ve Yorumlanabilirlik**: Model iç yapısı (örneğin, karar ağaçları veya regresyon katsayıları) incelenerek nasıl karar verdiği anlaşılabilir.  
    - **Düşük Hesaplama Maliyeti**: Genellikle derin öğrenme yöntemlerine kıyasla daha az kaynak gerektirir ve küçük–orta ölçekli veri setlerinde hızlı sonuç verir.  
    - **Özellik Mühendisliğinin Önemi**: Başarı, büyük ölçüde doğru ve anlamlı özelliklerin seçilmesine bağlıdır.  
    - **Çeşitli Algoritma Aileleri**:
        - **Doğrusal Modeller** (Logistic Regression)  
        - **Komşuluk Tabanlı Modeller** (KNN – K-Nearest Neighbors)  
        - **Ağaç Tabanlı Modeller** (Random Forest)  
        - **Kernel Tabanlı Modeller** (SVM – Support Vector Machine)  
        - **Topluluk Yöntemleri** (XGBoost)

    Bu modeller, etiketli verilerde sınıflandırma veya regresyon yapmak için ideal olup, veri miktarı ve boyutu arttıkça özellik mühendisliği, parametrik ayarlar ve model seçimi kritik hale gelir.
    """, "black")

    st.write("")

    model = st.selectbox("**Modeller Hakkında Detaylı Bilgi İçin Seçim Yapın**", [
        "Geleneksel Makine Öğrenme Modeli Seçin",
        "KNN",
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "XGBoost"
    ])

    ## KNN ##
    if model == "Geleneksel Makine Öğrenme Modeli Seçin":
        pass
    elif model == "KNN":
        st.header("KNN Model Sonuçları")
        with st.expander("Model Açıklaması"):
            st.markdown("""
        ### KNN Nedir?
        KNN, etiketli bir eğitim veri kümesindeki en yakın “K” komşuyu baz alarak yeni bir örneğin sınıfını tahmin eden **denetimli öğrenme** yöntemidir.  
        Veri uzayında bir noktanın etrafındaki K tane en yakın komşunun hangi sınıfa ait olduğuna bakarak, çoğunluğa göre karar verilir.  
        Örneğin **K=3** ise, en yakın üç komşunun ikisi “Pozitif” diğeriyse “Negatif” ise yeni nokta “Pozitif” sınıfına atanır.

        ### Çalışma Prensibi
        1. Eğitim aşamasında veri yalnızca saklanır, model oluşturma veya parametre öğrenme yapılmaz.  
        2. Tahmin aşamasında, genellikle **Öklidyen mesafe** kullanılarak sorgu noktasına en yakın K örnek bulunur.  
        3. Bu komşuların etiketleri arasındaki en yoğun sınıf yeni örneğe atanır.  
        4. **K** değeri, modelin genelleme yeteneğini belirleyen kritik bir hiperparametredir; küçük K gürültüye, büyük K ise sınır bölgelerine duyarsızlığa yol açabilir.

        ### Avantajları ve Dezavantajları
        - **Avantajları:**  
        - Uygulaması çok basit ve non-parametric (parametrik olmayan).  
        - Veri dağılımı hakkında ön kabul gerektirmez.  
        - **Dezavantajları:**  
        - Tahmin sırasında tüm veri kümesine bakarak performansı düşürebilir.  
        - Özelliklerin ölçü birimleri farklı ise mesafe hesaplamasında yanıltıcı olabilir; bu yüzden **ölçeklendirme** önemlidir.

        ### Kaynakça
        1. Altman, N. S. (1992). *An introduction to kernel and nearest‐neighbor nonparametric regression*. The American Statistician, 46(3), 175–185.
        """)
        st.write("")  # boşluk

        # 1) Hiperparametreler & 2) Test Doğruluğu

        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("1) En İyi Hiperparametreler")
            st.write("GridSearchCV ile elde edilen en iyi parametreler:")
            st.json({
                "metric": "manhattan",
                "n_neighbors": 4,
                "weights": "uniform"
            })
        with col2:
            st.subheader("2) Test Seti Doğruluğu")
            st.write("Test verisi üzerinde modelin doğruluk oranı:")
            st.metric(label="Accuracy", value="%95.61403508")
        st.write("---")  # yatay çizgi

        # 3) Classification Report & 4) Confusion Matrix
        rpt_col, cm_col = st.columns(2)
        with rpt_col:
            st.subheader("3) Eğitim Verisi Classification Report")
            st.write("Eğitim seti üzerindeki precision, recall, f1-score ve support değerleri:")
            report_df = pd.DataFrame({
                "precision": [0.95, 0.96, 0.96, 0.96],
                "recall":    [0.93, 0.97, 0.96, 0.95],
                "f1-score":  [0.94, 0.97, 0.96, 0.95],
                "support":   [42,   72,   114,  114]
            }, index=[
                "0 (Malignant)",
                "1 (Benign)",
                "accuracy",
                "macro avg"
            ])
            st.table(report_df)
        with cm_col:
            st.subheader("4) Test Verisi Confusion Matrix")
            st.write("Gerçek vs. tahmin sonuçlarının matrisi:")
            cm = np.array([[39, 3],
                        [ 2, 70]])
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(cm, interpolation='nearest')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(['Malignant (0)', 'Benign (1)'], rotation=45, ha='right')
            ax.set_yticklabels(['Malignant (0)', 'Benign (1)'])
            ax.set_xlabel("Tahminlenen Sınıf")
            ax.set_ylabel("Gerçek Sınıf")
            ax.set_title("KNN – Confusion Matrix")
            st.pyplot(fig)
        st.write("---")

        # 5) 10-Fold Çapraz Doğrulama
        st.subheader("5) 10-Fold Çapraz Doğrulama Sonuçları")
        st.write("Modelin 10 farklı katmandaki doğruluk değerleri ve ortalaması:")
        cv_scores = [
            0.98245614, 0.94736842, 0.94736842, 0.98245614, 1.0,
            1.0,        0.94736842, 0.98245614, 0.94736842, 0.96428571
        ]

        # Ortalama
        st.metric(label="Ortalama CV Doğruluk", value=f"{np.mean(cv_scores):.10%}")

        # Bar chart
        st.bar_chart(pd.DataFrame({"Fold Doğruluk": cv_scores}, index=[f"Fold {i}" for i in range(1, 11)]))
        

        st.write("")  # son boşluk

    ## ./KNN ##

    ## Logistic Regression ##
    elif model == 'Logistic Regression':
        st.header("Logistic Regression Model Sonuçları")
        with st.expander("Model Açıklaması"):
            st.markdown("""
            ### Logistic Regression Nedir?
            Logistic Regression, sonuçları 0 ile 1 arasında bir olasılık değeri olarak veren bir sınıflandırma yöntemidir. İkili sınıflandırma problemlerinde, bu olasılık üzerinden bir örneğin hangi sınıfa ait olacağına karar verilir.

            ### Çalışma Prensibi
            - Eğitim sırasında, model veriye bakarak her özelliğe bir ağırlık atar.  
            - Yeni bir örnekte, bu ağırlıkları kullanarak bir olasılık tahmini yapar.  
            - Tahmin 0.5’in üzerindeyse bir sınıf, altında ise diğer sınıf seçilir.  
            - Model parametreleri, tahminlerin veriyle en iyi uyuşmasını sağlayacak şekilde ayarlanır.

            ### Avantajları ve Dezavantajları
            - **Avantajları:**  
                - Sonuçları olasılık olarak sunar.  
                - Basit ve hızlıdır, küçük veri setlerinde etkili çalışır.  
                - Ağırlıkları inceleyerek hangi özelliklerin önemli olduğunu görebiliriz.  
            - **Dezavantajları:**  
                - Doğrusal sınırlamalar nedeniyle karmaşık ilişkileri yakalamada zorlanabilir.  
                - Dengesiz veri setlerinde yanıltıcı sonuçlara yol açabilir.  
                - Çok sayıda özellikle aşırı öğrenme riski bulunur.

            ### Kaynakça
            1. Menard, S. (2002). *Applied Logistic Regression Analysis*.  
            """)
        st.write("")  # boşluk

        # 1) Hiperparametreler & 2) Test Doğruluğu
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("1) En İyi Hiperparametreler")
            st.write("GridSearchCV ile elde edilen en iyi parametreler:")
            st.json({
                "C": 1,
                "penalty": "l2",
                "solver": "liblinear"
            })
        with col2:
            st.subheader("2) Test Seti Doğruluğu")
            st.write("Test verisi üzerinde modelin doğruluk oranı:")
            st.metric(label="Accuracy", value="%98.24561403")
        st.write("---")

        # 3) Classification Report & 4) Confusion Matrix
        rpt_col, cm_col = st.columns(2)
        with rpt_col:
            st.subheader("3) Eğitim Verisi Classification Report")
            st.write("Eğitim setindeki sınıf bazlı metrikler:")
            report_df = pd.DataFrame({
                "precision": [0.98, 0.99, 0.98, 0.98],
                "recall":    [0.98, 0.99, 0.98, 0.98],
                "f1-score":  [0.98, 0.99, 0.98, 0.98],
                "support":   [42,   72,   114,  114]
            }, index=[
                "0 (Malignant)",
                "1 (Benign)",
                "accuracy",
                "macro avg"
            ])
            st.table(report_df)

        with cm_col:
            st.subheader("4) Test Seti Confusion Matrix")
            st.write("Gerçek vs. tahmin sonuçlarının matrisi:")
            cm = np.array([[41, 1],
                        [ 1, 71]])
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(cm, interpolation='nearest')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(['Malignant (0)', 'Benign (1)'], rotation=45, ha='right')
            ax.set_yticklabels(['Malignant (0)', 'Benign (1)'])
            ax.set_xlabel("Tahminlenen Sınıf")
            ax.set_ylabel("Gerçek Sınıf")
            ax.set_title("LR – Confusion Matrix")
            st.pyplot(fig)
        st.write("---")

        # 5) 10-Fold Çapraz Doğrulama Sonuçları
        st.subheader("5) 10-Fold Çapraz Doğrulama Sonuçları")
        st.write("Modelin 10 farklı katmandaki doğruluk değerlerinin dağılımı ve ortalaması:")
        cv_scores = [
            1.0000000000, 1.0000000000, 0.9347826087, 0.9782608696, 1.0000000000,
            0.9777777778, 1.0000000000, 0.9777777778, 0.9333333333, 0.9777777778
        ]

        # Ortalama
        st.metric(label="Ortalama CV Doğruluk", value=f"{np.mean(cv_scores):.10%}")

        # Bar chart
        st.bar_chart(
            pd.DataFrame({"Fold Doğruluk": cv_scores},
                        index=[f"Fold {i}" for i in range(1, 11)])
        )
        st.write("")  # son boşluk


    ## ./Logistic Regression ##    


    ## Random Forest ##
    elif model == 'Random Forest':
        st.header("Random Forest Model Sonuçları")
        with st.expander("Model Açıklaması"):
            st.markdown("""
            ### Random Forest Nedir?
            Random Forest, birden fazla karar ağacını bir araya getirerek (ensemble) tahmin yapan bir algoritmadır. Her ağaç, eğitim verisinin farklı bir alt örneği üzerinde eğitilir ve yeni bir örnek için tüm ağaçların oy çokluğuna göre sınıf kararını belirler.

            ### Çalışma Prensibi
            - Eğitim sırasında, veri kümesinden rastgele alt kümeler (bootstrap) oluşturulur ve her alt küme ayrı bir karar ağacına öğretilir.  
            - Ağaçlar farklı özellik alt kümleri kullanılarak bölünme noktaları seçer, böylece çeşitlilik sağlanır.  
            - Tahmin aşamasında, her ağacın verdiği sınıf oyları toplanır ve en çok oyu alan sınıf nihai tahmin olarak seçilir.

            ### Avantajları ve Dezavantajları
            - **Avantajları:**  
                - Aşırı öğrenme (overfitting) riskini azaltır.  
                - Farklı veri ve özellik alt kümeleriyle güçlü ve kararlı modeller üretir.  
                - Özellik önemini (feature importance) hesaplayarak hangi değişkenlerin etkili olduğunu gösterir.  
            - **Dezavantajları:**  
                - Çok sayıda ağaç nedeniyle hesaplama maliyeti ve bellek kullanımı yükselebilir.  
                - Tek tek karar ağaçlarına göre yorumlanabilirliği düşüktür.

            ### Kaynakça
            1. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.   
            """)
        st.write("")  # boşluk

        # 1) Hiperparametreler & 2) Test Doğruluğu
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("1) En İyi Hiperparametreler")
            st.write("GridSearchCV ile elde edilen en iyi parametreler:")
            st.json({
                "n_estimators": 100,
                "max_depth": None,
                "max_features": 0.5,
                "min_samples_split": 2,
                "min_samples_leaf": 2
            })
        with col2:
            st.subheader("2) Test Seti Doğruluğu")
            st.write("Test verisi üzerinde modelin doğruluk oranı:")
            st.metric(label="Accuracy", value="%95.61403509")
        st.write("---")

        # 3) Classification Report & 4) Confusion Matrix
        rpt_col, cm_col = st.columns(2)
        with rpt_col:
            st.subheader("3) Eğitim Verisi Classification Report")
            st.write("Eğitim setindeki sınıf bazlı metrikler:")
            report_df = pd.DataFrame({
                "precision": [0.95, 0.96, 0.96, 0.96],
                "recall":    [0.93, 0.97, 0.96, 0.95],
                "f1-score":  [0.94, 0.97, 0.96, 0.95],
                "support":   [42,   72,   114,  114]
            }, index=[
                "Benign (1)",
                "Malignant (0)",
                "accuracy",
                "macro avg"
            ])
            st.table(report_df)

        with cm_col:
            st.subheader("4) Test Seti Confusion Matrix")
            st.write("Gerçek vs. tahmin sonuçlarının matrisi:")
            cm = np.array([[39, 3],
                        [ 2, 70]])
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(cm, interpolation='nearest')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(['Benign (1)', 'Malignant (0)'], rotation=45, ha='right')
            ax.set_yticklabels(['Benign (1)', 'Malignant (0)'])
            ax.set_xlabel("Tahminlenen Sınıf")
            ax.set_ylabel("Gerçek Sınıf")
            ax.set_title("RF – Confusion Matrix")
            st.pyplot(fig)
        st.write("---")

        # 5) 10-Fold Çapraz Doğrulama Sonuçları
        st.subheader("5) 10-Fold Çapraz Doğrulama Sonuçları")
        st.write("Modelin 10 farklı katmandaki doğruluk değerlerinin dağılımı ve ortalaması:")
        cv_scores = [
            0.9130434783,
            1.0000000000,
            0.9565217391,
            0.9782608696,
            0.9565217391,
            0.9777777778,
            1.0000000000,
            0.9777777778,
            0.8888888889,
            0.9111111111
        ]
        # Ortalama
        st.metric(label="Ortalama CV Doğruluk", value=f"{np.mean(cv_scores):.10%}")
        # Bar chart
        st.bar_chart(
            pd.DataFrame({"Fold Doğruluk": cv_scores},
                        index=[f"Fold {i}" for i in range(1, 11)])
        )
        
        st.write("")  # son boşluk
    ## ./Random Forest ##

    ## SVM ##
    elif model == "SVM":
        st.header("SVM Model Sonuçları")
        with st.expander("Model Açıklaması"):
            st.markdown("""
            ### SVM Nedir?
            Support Vector Machine (SVM), veriyi doğru sınıflandırmak için en geniş marjini (sınıflar arası boşluğu) bulmaya çalışan bir denetimli öğrenme algoritmasıdır. Hem ikili hem de çok sınıflı sınıflandırma problemlerine uygulanabilir.

            ### Çalışma Prensibi
            - Eğitim verisindeki örnekler, özellik uzayında birer nokta olarak kabul edilir.  
            - SVM, bu noktaları ayıracak en geniş sınır çizgisini (veya hiper düzlemi) belirler.  
            - Sınır çizgisine en yakın veri noktaları “destek vektörleri” olarak adlandırılır ve marjini belirlemede kritik rol oynar.  
            - Karmaşık veriler için çekirdek (kernel) yöntemiyle doğrusal olmayan sınırlamalar da oluşturulabilir.

            ### Avantajları ve Dezavantajları
            - **Avantajları:**  
                - Marjin maksimize edilerek genelleme hatası azaltılır.  
                - Kernel fonksiyonlarıyla doğrusal olmayan sorunları da çözebilir.  
                - Özellik sayısı yüksek veri setlerinde etkili çalışır.  
            - **Dezavantajları:**  
                - Büyük ölçekli veri setlerinde eğitim süresi ve bellek kullanımı artar.  
                - Doğru kernel ve hiperparametre seçimi uzmanlık gerektirir.  
                - Çıktısı doğrudan olasılık olmadığı için yorumlama ek adım isteyebilir.

            ### Kaynakça
            1. Cortes, C., & Vapnik, V. (1995). *Support-Vector Networks*. Machine Learning, 20(3), 273–297.  
            """)

        st.write("")  # boşluk

        # 1) En İyi Hiperparametreler & 2) Test Seti Doğruluğu
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("1) En İyi Hiperparametreler")
            st.write("GridSearchCV ile elde edilen en iyi parametreler:")
            st.json({
                "C": 0.1,
                "kernel": "linear",
                "gamma": "scale"
            })
        with col2:
            st.subheader("2) Test Seti Doğruluğu")
            st.write("Test verisi üzerindeki doğruluk oranı:")
            st.metric(label="Accuracy", value="%98.24561403")
        st.write("---")

        # 3) Classification Report & 4) Confusion Matrix
        rpt_col, cm_col = st.columns(2)
        with rpt_col:
            st.subheader("3) Eğitim Verisi Classification Report")
            st.write("Eğitim seti üzerindeki sınıf bazlı metrikler:")
            report_df = pd.DataFrame({
                "precision": [0.98, 0.99, 0.98, 0.98],
                "recall":    [0.98, 0.99, 0.98, 0.98],
                "f1-score":  [0.98, 0.99, 0.98, 0.98],
                "support":   [42,   72,   114, 114]
            }, index=[
                "0 (Malignant)",
                "1 (Benign)",
                "accuracy",
                "macro avg"
            ])
            st.table(report_df)

        with cm_col:
            st.subheader("4) Test Seti Confusion Matrix")
            st.write("Gerçek vs. tahmin sonuçlarının matrisi:")
            cm = np.array([[41, 1],
                        [ 1, 71]])
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(cm, interpolation='nearest')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Malignant (0)', 'Benign (1)'], rotation=45, ha='right')
            ax.set_yticklabels(['Malignant (0)', 'Benign (1)'])
            ax.set_xlabel("Tahminlenen Sınıf")
            ax.set_ylabel("Gerçek Sınıf")
            ax.set_title("SVM – Confusion Matrix")
            st.pyplot(fig)
        st.write("---")

        # 5) 10-Fold Çapraz Doğrulama Sonuçları
        st.subheader("5) 10-Fold Çapraz Doğrulama Sonuçları")
        st.write("Modelin her fold’daki doğruluk skorları ve ortalaması:")
        cv_scores = [
            0.9824561404,
            0.9649122807,
            0.9824561404,
            0.9824561404,
            1.0000000000,
            0.9824561404,
            0.9298245614,
            1.0000000000,
            1.0000000000,
            0.9642857143
        ]


        # Ortalama skoru göster
        st.metric(label="Ortalama CV Doğruluk", value=f"{np.mean(cv_scores):.10%}")

        # Bar chart
        st.bar_chart(
            pd.DataFrame({"Fold Doğruluk": cv_scores},
                        index=[f"Fold {i}" for i in range(1, 11)])
        )
        
        st.write("")  # son boşluk
    ## ./SVM ##

    ## XGBOOST ##
    elif model == "XGBoost":
        st.header("XGBoost Model Sonuçları")

        # Açıklama expander'ı
        with st.expander("Model Açıklaması"):
            st.markdown("""
            ### XGBoost Nedir?
            XGBoost (eXtreme Gradient Boosting), ağaç tabanlı zayıf öğrenicileri ardışık olarak eğiterek hataları azaltan, yüksek performanslı bir topluluk (ensemble) algoritmasıdır. Gradient Boosting altyapısını optimize ederek hız, bellek kullanımı ve genelleme yeteneği konusunda iyileştirmeler sunar.

            ### Çalışma Prensibi
            - **Artımlı Ağaç Eğitimi:** Her seferinde bir karar ağacı eklenir; yeni ağaç, önceki ağaçların yaptığı hataları düzeltecek şekilde eğitilir.  
            - **Regularizasyon:** Ağaç derinliği ve yaprak sayısı sınırlandırılarak aşırı öğrenme (overfitting) riski azaltılır.  
            - **Paralel Eğitim:** Dalların bulunması (split finding) paralel olarak gerçekleştirilerek eğitim hızı artırılır.  
            - **Eksik Değer İşleme:** Özelliklerdeki boş değerler otomatik olarak yönlendirilir; ek ön işleme gerek kalmaz.

            ### Avantajları ve Dezavantajları
            - **Avantajları:**  
                - Yüksek tahmin doğruluğu ve genelleme gücü  
                - Bellek ve işlem optimizasyonu sayesinde büyük veri setlerinde hızlı çalışır  
                - Eksik veri ve kategorik değişkenlerle esnek başa çıkma  
                - Hiperparametrelerle ayrıntılı kontrol ve düzenleme imkânı  
            - **Dezavantajları:**  
                - Çok sayıda hiperparametre ayarı uzmanlık gerektirir  
                - Modelin yorumlanabilirliği tek ağaçlara göre düşüktür  
                - Aşırı büyük veri setlerinde bellek kullanımı hala yüksek olabilir

            ### Kaynakça
            1. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*.
            """)

        # 1) En İyi Hiperparametreler & 2) Test Seti Doğruluğu
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("1) En İyi Hiperparametreler")
            st.json({
                "learning_rate": 0.1,
                "max_depth": 4,
                "n_estimators": 100,
                "subsample": 0.8
            })
        with col2:
            st.subheader("2) Test Seti Doğruluğu")
            st.metric(label="Accuracy", value="%96.49122807")
        st.write("---")

        # 3) Classification Report & 4) Confusion Matrix
        rpt_col, cm_col = st.columns(2)
        with rpt_col:
            st.subheader("3) Eğitim Verisi Classification Report")
            report_df = pd.DataFrame({
                "precision": [0.97, 0.96, 0.96, 0.97],
                "recall":    [0.93, 0.99, 0.96, 0.96],
                "f1-score":  [0.95, 0.97, 0.96, 0.96],
                "support":   [42,   72,   114, 114]
            }, index=[
                "0 (Malignant)",
                "1 (Benign)",
                "accuracy",
                "macro avg"
            ])
            st.table(report_df)

        with cm_col:
            st.subheader("4) Test Seti Confusion Matrix")
            cm = np.array([[39, 3],
                        [ 1, 71]])
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(cm, interpolation='nearest')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(['Malignant (0)', 'Benign (1)'], rotation=45, ha='right')
            ax.set_yticklabels(['Malignant (0)', 'Benign (1)'])
            ax.set_xlabel("Tahminlenen Sınıf")
            ax.set_ylabel("Gerçek Sınıf")
            ax.set_title("XGB – Confusion Matrix")
            st.pyplot(fig)
        st.write("---")

        # 5) 10-Fold Çapraz Doğrulama Sonuçları
        st.subheader("5) 10-Fold Çapraz Doğrulama Sonuçları")
        cv_scores = [
            0.9565217391,
            0.9782608696,
            0.9782608696,
            0.9782608696,
            0.9782608696,
            0.9777777778,
            1.0000000000,
            0.9777777778,
            0.8444444444,
            0.9333333333
        ]

        st.metric(label="Ortalama CV Doğruluk", value=f"{np.mean(cv_scores):.10%}")

        st.bar_chart(
            pd.DataFrame({"Fold Doğruluk": cv_scores},
                        index=[f"Fold {i}" for i in range(1, 11)])
        )
    ## ./XGBOOST ##

    st.write('---')

    ## Toplu Degerlendirme ##
    models = [
    "KNN",
    "Logistic Regression",
    "Random Forest",
    "SVM",
    "XGBoost"
    ]
    
    accuracies = [
        97.0112781000,
        97.7971014500,
        95.5990338170,
        97.8884711800,
        96.0289855080
    ]

    # Tek bir DataFrame oluşturuluyor
    df = pd.DataFrame({
        "Model": models,
        "Accuracy (%)": accuracies
    })

    st.header("Geleneksel Modellerin Ortalama Doğruluk Oranları")

    col1, col2 = st.columns(2)

    with col1:   
        # Mevcut df’inizi Accuracy (%) sütununa göre büyükten küçüğe sıralama
        df_sorted = df.sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)

        # Streamlit içinde gösterme
        st.table(df_sorted)

    with col2:
        # Renkli çubuk grafik
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Model:N", title="Model"),
                y=alt.Y("Accuracy (%):Q", title="Accuracy (%)"),
                color=alt.Color("Model:N", legend=None)
            )
            .properties(width="container", height=400)
        )

        st.altair_chart(chart, use_container_width=True)

    # Açıklama
    st.subheader("Sonuç")
    boxInfo("""
    Breast Cancer Wisconsin (Diagnostic) veri seti; tümörlerin **malignant** (kötü huylu) veya **benign** (iyi huylu) olarak sınıflandığı, özellik çıkarımına dayalı bir sınıflandırma problemidir. Burada beş farklı geleneksel makine öğrenmesi algoritmasının ortalama doğruluk oranları karşılaştırılmıştır:

    - **SVM (97.89 %)** ve **Logistic Regression (97.80 %)**, en yüksek doğruluk değerlerini elde ederek veri setindeki sınırların büyük ölçüde lineer veya kernel tabanlı yöntemlerle ayrıştığını gösteriyor.  
    - **KNN (97.01 %)** de oldukça yüksek bir başarı sergiliyor; bu, komşuluk temelli basit bir yöntemin dahi bu problemde etkili olabileceğini ortaya koyuyor.  
    - **XGBoost (96.03 %)** ve **Random Forest (95.60 %)**, ağaç tabanlı ensemble yöntemler olmalarına rağmen görece bir miktar daha düşük performans gösteriyor. Bu, veri setindeki özelliklerin “çok derin” veya “karmaşık” karar sınırlarına ihtiyaç duymadığına işaret edebilir.

    Sonuç olarak, veri seti hem doğrusal hem de kernel tabanlı yaklaşımlara çok uygun; bu da Logistic Regression ve SVM’in yüksek performansını açıklıyor. Ağaç tabanlı yöntemlerin biraz geride kalması, muhtemelen ek özellik mühendisliği veya hiperparametre optimizasyonu gereksiniminden kaynaklanabilir. Ancak tüm modellerin %95’in üzerinde performans göstermesi, veri setinin genel anlamda temiz, dengeli ve ayırt edici özellikler içerdiğini doğruluyor. Sınıflandırma başarısının kritik olduğu bir tıbbi uygulamada, **hassasiyet** ve **özgüllük** gibi ek metrikler de incelenerek en uygun model seçilmelidir.
    """, "green")

    

    ## ./Toplu Degerlendirme ##

    st.write("")
    st.write("---")
    st.write("")

    st.title("Derin Öğrenme Modelleri")
    boxExp("""
    Derin öğrenme modelleri, büyük ölçekli ve yüksek boyutlu veri setleri için idealdir. Bu modeller, özellikleri veriden otomatik olarak öğrenir ve genellikle karmaşık yapılandırmalar içerir. Yaygın özellikleri şunlardır:

    - **Otomatik Özellik Çıkarımı**: Verilerden anlamlı özellikleri otomatik olarak çıkarır.
    - **Yüksek Hesaplama Gücü Gereksinimi**: Daha fazla hesaplama gücü ve zaman gerektirir.
    - **Yüksek Performans**: Karmaşık veri setlerinde üstün performans gösterir.
    - **Popüler Ağ Yapıları**:
        - **Evrişimli Sinir Ağları (CNN)**  
        - **Uzun Kısa Vadeli Bellek Ağları (LSTM)**
        - **Çok Katmanlı Algılayıcılar (MLP)**: Düz bağlantılı katmanlardan oluşan yapısıyla genel amaçlı sınıflandırma ve regresyon problemlerinde kullanılır.
        - **TabNet**: Özellikle tabular (yapısal) verilerde dikkat mekanizması ile karar odaklı öğrenme yapan, yorumlanabilir ve güçlü bir modeldir.

    Bu modeller görüntü, ses, metin gibi karmaşık verileri işlerken oldukça başarılıdır. Ayrıca TabNet gibi bazı modeller, yapılandırılmış (tabular) veri setlerinde geleneksel modellere kıyasla yüksek doğruluk ve yorumlanabilirlik sağlar.
    """, "black")

    st.write("")

    model = st.selectbox("**Modeller Hakkında Detaylı Bilgi İçin Seçim Yapın**", [
        "Derin Öğrenme Modeli Seçiniz",
        "CNN",
        "LSTM",
        "MLP",
        "TabNet"
    ])

    ## CNN ##
    if model == "Derin Öğrenme Modeli Seçiniz":
        pass
    elif model == "CNN":
        st.header("CNN Model Sonuçları")
        with st.expander("Model Açıklaması"):
            st.markdown("""
            ### CNN Nedir?
            CNN, özellikle görüntü, ses ya da zaman serisi gibi verilerde otomatik olarak anlamlı özellikleri çıkaran bir derin öğrenme modelidir. Katman katman çalışan yapısıyla ham veriden doğrudan öğrenme yapar.

            ### Çalışma Prensibi
            - **Konvolüsyon Katmanları:** Veri üzerinde kaydırılan küçük filtreler (kernel’ler) ile yerel özellik haritaları çıkarır.  
            - **Havuzlama (Pooling) Katmanları:** Özellik haritalarını küçülterek işlem yükünü ve aşırı öğrenmeyi azaltır.  
            - **Tam Bağlantılı Katmanlar:** Son katmanlarda elde edilen yüksek seviyeli özellikleri kullanarak sınıflandırma veya regresyon görevini yapar.  
            - **Geri Yayılım (Backpropagation):** Hata hesaplanıp katman ağırlıkları güncellenerek model eğitilir.

            ### Avantajları ve Dezavantajları
            - **Avantajları:**  
                - Özellik mühendisliği gerektirmeden ham veriden öğrenir.  
                - Görüntü ve ses gibi yapılandırılmamış verilerde çok başarılıdır.  
                - Parametre paylaşımı sayesinde daha az ağırlık ile güçlü modeller oluşturur.  
            - **Dezavantajları:**  
                - GPU ve büyük veri gereksinimi vardır; eğitim maliyeti yüksektir.  
                - Hiperparametre (katman sayısı, filtre boyutu vb.) ayarı uzmanlık ister.  
                - Yorumlanabilirliği sınırlıdır; hangi filtrelerin ne öğrendiğini görmek zordur.

            ### Kaynakça
            1. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet classification with deep convolutional neural networks*.  
            """)

        st.write("")  # boşluk

        # 1) En İyi Hiperparametreler & 2) Test Doğruluğu
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("1) En İyi Hiperparametreler")
            st.write("GridSearchCV ile elde edilen en iyi parametreler:")
            st.json({
                "batch_size": 16,
                "epochs": 100,
                "activation": "relu",
                "dropout_rate": 0.4,
                "filters": 32,
                "kernel_size": 3,
                "learning_rate": 0.001
            })
        with col2:
            st.subheader("2) Test Seti Doğruluğu")
            st.write("Test verisi üzerinde modelin doğruluk oranı:")
            st.metric(label="Accuracy", value="%98.24561404")
        st.write("---")

        # 3) Classification Report & 4) Confusion Matrix
        rpt_col, cm_col = st.columns(2)
        with rpt_col:
            st.subheader("3) Test Seti Classification Report")
            st.write("Test setindeki sınıf bazlı metrikler:")
            report_df = pd.DataFrame({
                "precision": [0.97, 1.00, 0.99, 0.99],
                "recall":    [1.00, 0.95, 0.98, 0.98],
                "f1-score":  [0.99, 0.98, 0.98, 0.98],
                "support":   [72,   42,   114,  114]
            }, index=[
                "0 (Malignant)",
                "1 (Benign)",
                "accuracy",
                "macro avg"
            ])
            st.table(report_df)

        with cm_col:
            st.subheader("4) Test Seti Confusion Matrix")
            st.write("Gerçek vs. tahmin sonuçlarının matrisi:")

            cm = np.array([[72, 0], [2, 40]])
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14, color="white" if cm[i, j] > cm.max()/2 else "black")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Malignant (0)', 'Benign (1)'], rotation=45, ha='right')
            ax.set_yticklabels(['Malignant (0)', 'Benign (1)'])
            ax.set_xlabel("Tahmin Edilen Sınıf")
            ax.set_ylabel("Gerçek Sınıf")
            ax.set_title("CNN – Confusion Matrix")
            st.pyplot(fig)
        st.write("---")

        # 5) 10-Fold Çapraz Doğrulama Sonuçları
        st.subheader("5) 10-Fold Çapraz Doğrulama Sonuçları")
        st.write("Modelin 10 farklı katmandaki doğruluk değerlerinin dağılımı ve ortalaması:")
        cv_scores = [
            0.9782608696, 0.9565217391, 0.9565217391, 0.9782608696, 1.0,
            0.9777777778, 1.0, 0.9555555556, 0.9333333333, 1.0
        ]

        # Ortalama
        st.metric(label="Ortalama CV Doğruluk", value=f"{np.mean(cv_scores):.10%}")

        # Bar chart
        st.bar_chart(
            pd.DataFrame({"Fold Doğruluk": cv_scores},
                        index=[f"Fold {i}" for i in range(1, 11)])
        )

        st.write("")  # son boşluk

    ## ./ CNN ##

    ## LSTM ##
    elif model == "LSTM":
        st.header("LSTM Model Sonuçları")
        with st.expander("Model Açıklaması"):
            st.markdown("""
            ### LSTM Nedir?
            Long Short-Term Memory (LSTM), özellikle zaman serisi ve metin gibi sıralı verilerde uzun vadeli bağımlılıkları öğrenebilmek için geliştirilmiş bir tekrarlayan sinir ağı (RNN) hücresidir. Diğer RNN’lerin yaşadığı **vanishing gradient** sorununu azaltarak uzun dizilerdeki bilgiyi korur.

            ### Çalışma Prensibi
            - **Giriş Kapısı (Input Gate):** Yeni bilgiyi hücre durumuna ne oranda ekleyeceğini kontrol eder.  
            - **Unutma Kapısı (Forget Gate):** Önceki hücre durumundan ne kadarının saklanacağını belirler.  
            - **Çıkış Kapısı (Output Gate):** Hücre durumundan hangi bilginin çıkış katmanına aktarılacağını seçer.  
            - Bu üçlü kapı mekanizması sayesinde hem eski bilgiyi unutmadan hem de gereksiz detayları atarak öğrenme yapılır.

            ### Avantajları ve Dezavantajları
            - **Avantajları:**  
                - Uzun vadeli bağımlılıkları yakalayarak metin ve zaman serisi problemlerinde başarı sağlar.  
                - Vanishing gradient etkisini azaltır.  
            - **Dezavantajları:**  
                - Standart RNN’lere göre daha fazla parametre içerdiğinden eğitim süresi uzundur.  
                - Model karmaşıklığı arttıkça overfitting riski doğabilir.

            ### Kaynakça
            1. Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). *Learning to forget: Continual prediction with LSTM*. Neural Computation, 12(10), 2451–2471.   
            """)

        st.write("")  # boşluk

        # 1) En İyi Hiperparametreler & 2) Test Doğruluğu
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("1) En İyi Hiperparametreler")
            st.write("GridSearchCV ile elde edilen en iyi parametreler:")
            st.json({
                "batch_size": 16,
                "epochs": 20,
                "dropout_rate": 0.3,
                "units_1": 64,
                "units_2": 32
            })
        with col2:
            st.subheader("2) Test Seti Doğruluğu")
            st.write("Test verisi üzerinde modelin doğruluk oranı:")
            st.metric(label="Accuracy", value="%97.79975601")
        st.write("---")

        # 3) Classification Report & 4) Confusion Matrix
        rpt_col, cm_col = st.columns(2)
        with rpt_col:
            st.subheader("3) Test Seti Classification Report")
            st.write("Test setindeki sınıf bazlı metrikler:")
            report_df = pd.DataFrame({
                "precision": [0.99, 0.95, 0.97, 0.97],
                "recall":    [0.97, 0.98, 0.97, 0.97],
                "f1-score":  [0.98, 0.97, 0.97, 0.97],
                "support":   [71, 43, 114, 114]
            }, index=[
                "0 (Malignant)",
                "1 (Benign)",
                "accuracy",
                "macro avg"
            ])
            st.table(report_df)

        with cm_col:
            st.subheader("4) Test Seti Confusion Matrix")
            st.write("Gerçek vs. tahmin sonuçlarının matrisi:")

            cm = np.array([[69, 2], [1, 42]])
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14,
                            color="white" if cm[i, j] > cm.max()/2 else "black")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Malignant (0)', 'Benign (1)'], rotation=45, ha='right')
            ax.set_yticklabels(['Malignant (0)', 'Benign (1)'])
            ax.set_xlabel("Tahmin Edilen Sınıf")
            ax.set_ylabel("Gerçek Sınıf")
            ax.set_title("LSTM – Confusion Matrix")
            st.pyplot(fig)
        st.write("---")

        # 5) 10-Fold Çapraz Doğrulama Sonuçları
        st.subheader("5) 10-Fold Çapraz Doğrulama Sonuçları")
        st.write("Modelin 10 farklı katmandaki doğruluk değerlerinin dağılımı ve ortalaması:")
        cv_scores = [
            0.96491228, 0.98245614, 1.0, 0.94736842, 0.98245614,
            0.94736842, 1.0, 0.96491228, 0.98245614, 1.0
        ]

        # Ortalama
        st.metric(label="Ortalama CV Doğruluk", value=f"{np.mean(cv_scores):.10%}")

        # Bar chart
        st.bar_chart(
            pd.DataFrame({"Fold Doğruluk": cv_scores},
                        index=[f"Fold {i}" for i in range(1, 11)])
        )

        st.write("")  # son boşluk
    
    ## ./LSTM ##

    ## MLP ##
    elif model == 'MLP':
        st.header("MLP (Multilayer Perceptron) Model Sonuçları")
        with st.expander("Model Açıklaması"):
            st.markdown("""
            ### MLP Nedir?
            Multi-Layer Perceptron (MLP), birden fazla tam bağlantılı katmandan (input, bir veya daha fazla hidden, output) oluşan yapay sinir ağı modelidir. Genellikle sınıflandırma ve regresyon problemlerinde kullanılır.

            ### Çalışma Prensibi
            - **Girdi Katmanı:** Ham özellikler sinir ağına beslenir.  
            - **Gizli Katmanlar:** Her düğüm (nöron), bir önceki katmandan gelen ağırlıklı toplamı alır ve genellikle `ReLU`, `tanh` veya `sigmoid` gibi bir aktivasyon fonksiyonundan geçirir.  
            - **Çıkış Katmanı:** Son katmandaki nöronlar, problemin türüne göre sınıf olasılıklarını veya sürekli değer tahminlerini üretir.  
            - **Geri Yayılım (Backpropagation):** Hata, çıkıştan girdiye doğru katman katman iletilir ve her ağırlık, öğrenme oranına bağlı olarak güncellenir.

            ### Avantajları ve Dezavantajları
            - **Avantajları:**  
                - Esnek bir yapıya sahiptir; katman ve nöron sayısı kolayca ayarlanabilir.  
                - Doğrusal olmayan ilişkileri modelleyebilir.  
                - Geniş bir problem yelpazesinde (görüntü, metin, sayısal veri) kullanılabilir.  
            - **Dezavantajları:**  
                - Çok sayıda parametre içerir; overfitting riskini artırır.  
                - Hiperparametre (katman sayısı, nöron adedi, öğrenme hızı) ayarı uzmanlık gerektirir.  
                - Eğitim için genellikle GPU ve yüksek hesaplama gücü gerekir.

            ### Kaynakça
            1. Scikit-learn: Supervised neural network models (MLP).  
            https://scikit-learn.org/stable/modules/neural_networks_supervised.html
            """)

        st.write("")  # boşluk

        # 1) En İyi Hiperparametreler & 2) Test Doğruluğu
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("1) En İyi Hiperparametreler")
            st.write("GridSearchCV ile elde edilen en iyi parametreler:")
            st.json({
                "activation": "relu",
                "alpha": 0.0001,
                "hidden_layer_sizes": (100, 50),
                "learning_rate": "constant",
                "solver": "sgd"
            })
        with col2:
            st.subheader("2) Test Seti Doğruluğu")
            st.write("Test verisi üzerinde modelin doğruluk oranı:")
            st.metric(label="Accuracy", value="%96.49122807")
        st.write("---")

        # 3) Classification Report & 4) Confusion Matrix
        rpt_col, cm_col = st.columns(2)
        with rpt_col:
            st.subheader("3) Test Seti Classification Report")
            st.write("Test setindeki sınıf bazlı metrikler:")
            report_df = pd.DataFrame({
                "precision": [0.97, 0.95, 0.96, 0.96],
                "recall":    [0.97, 0.95, 0.96, 0.96],
                "f1-score":  [0.97, 0.95, 0.96, 0.96],
                "support":   [71, 43, 114, 114]
            }, index=[
                "0 (Malignant)",
                "1 (Benign)",
                "accuracy",
                "macro avg"
            ])
            st.table(report_df)

        with cm_col:
            st.subheader("4) Test Seti Confusion Matrix")
            st.write("Gerçek vs. tahmin sonuçlarının matrisi:")

            cm = np.array([[69, 2], [2, 41]])
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14,
                            color="white" if cm[i, j] > cm.max()/2 else "black")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Malignant (0)', 'Benign (1)'], rotation=45, ha='right')
            ax.set_yticklabels(['Malignant (0)', 'Benign (1)'])
            ax.set_xlabel("Tahmin Edilen Sınıf")
            ax.set_ylabel("Gerçek Sınıf")
            ax.set_title("MLP – Confusion Matrix")
            st.pyplot(fig)
        st.write("---")

        # 5) 10-Fold Çapraz Doğrulama Sonuçları
        st.subheader("5) 10-Fold Çapraz Doğrulama Sonuçları")
        st.write("Modelin 10 farklı katmandaki doğruluk değerlerinin dağılımı ve ortalaması:")
        cv_scores = [
            1.0, 0.96491228, 0.96491228, 0.96491228, 0.94736842,
            0.96491228, 1.0, 0.94736842, 0.98245614, 1.0
        ]

        st.metric(label="Ortalama CV Doğruluk", value=f"{np.mean(cv_scores):.10%}")

        st.bar_chart(
            pd.DataFrame({"Fold Doğruluk": cv_scores},
                        index=[f"Fold {i}" for i in range(1, 11)])
        )

        st.write("")  # son boşluk

    ## ./MLP ##

    ## TABNET ##
    elif model == 'TabNet':
        st.header("TabNet Model Sonuçları")
        with st.expander("Model Açıklaması"):
            st.markdown("""
            ### TabNet Nedir?
            TabNet, yapısal (tabular) veriler üzerinde çalışan ve karar adımları ile dikkat mekanizmasını birleştiren bir derin öğrenme modelidir. Geleneksel tekil ağlarda olduğu gibi ham veri üzerinden anlamlı özellikler öğrenir, ancak her adımda hangi özelliklerin kullanılacağını dinamik olarak seçer.

            ### Çalışma Prensibi
            - **Karar Adımları (Decision Steps):** Model, veriyi ardışık bir dizi karar adımından geçirir ve her adımda seçilen özelliklere odaklanır.  
            - **Dikkat Mekanizması (Feature Selection):** Attention ağırlıkları sayesinde her karar adımında en önemli özellikler belirlenir.  
            - **Geri Besleme:** Bir adımda öğrenilen özellikler, sonraki adımların girdisi olarak kullanılarak bilgi akışı sağlanır.  
            - **Sıfırdan Öğrenme:** Model, her adımda farklı özellik alt kümeleriyle çalışarak yorumlanabilir bir yapı oluşturur.

            ### Avantajları ve Dezavantajları
            - **Avantajları:**  
            - Tabular verilerde yüksek doğruluk ve genelleme yeteneği.  
            - Klinik veya finansal gibi hassas alanlarda yorumlanabilirlik sunar.  
            - Özellik seçimi dahili olarak yapıldığı için ek ön işleme gerek kalmaz.  
            - **Dezavantajları:**  
            - Karmaşık mimarisi nedeniyle eğitim süresi ve bellek kullanımı yüksektir.  
            - Hiperparametre sayısı ve karar adımı derinliği ayarı uzmanlık gerektirir.  
            - Küçük veri setlerinde aşırı öğrenme riski olabilir.

            ### Kaynakça
            1. Arik, S. Ö., & Pfister, T. (2019). *TabNet: Attentive Interpretable Tabular Learning*. arXiv preprint arXiv:1908.07442.  
            2. PyTorch TabNet Documentation: https://dreamquark-ai.github.io/tabnet/
            """)

        st.write("")  # boşluk

        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("1) En İyi Hiperparametreler")
            st.write("Modelin elle ayarlanmış en iyi hiperparametre değerleri:")
            st.json({
                "n_d": 8,
                "n_a": 16,
                "n_steps": 3,
                "gamma": 1.5,
                "lambda_sparse": 0.0001,
                "seed": 42,
                "verbose": 0
            })

        with col2:
            # 1) Test Seti Doğruluğu
            st.subheader("1) Test Seti Doğruluğu")
            st.write("Test verisi üzerinde modelin doğruluk oranı:")
            st.metric(label="Accuracy", value="%100.00000000")
            
        st.write("---")
        # 2) Classification Report & 3) Confusion Matrix
        rpt_col, cm_col = st.columns(2)
        with rpt_col:
            st.subheader("2) Test Seti Classification Report")
            st.write("Test setindeki sınıf bazlı metrikler:")
            report_df = pd.DataFrame({
                "precision": [1.00, 1.00, 1.00, 1.00],
                "recall":    [1.00, 1.00, 1.00, 1.00],
                "f1-score":  [1.00, 1.00, 1.00, 1.00],
                "support":   [35,   21,   56,   56]
            }, index=[
                "0 (Malignant)",
                "1 (Benign)",
                "accuracy",
                "macro avg"
            ])
            st.table(report_df)

        with cm_col:
            st.subheader("3) Test Seti Confusion Matrix")
            st.write("Gerçek vs. tahmin sonuçlarının matrisi:")
            cm = np.array([[35, 0], [0, 21]])
            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14,
                            color="white" if cm[i, j] > cm.max()/2 else "black")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Malignant (0)', 'Benign (1)'], rotation=45, ha='right')
            ax.set_yticklabels(['Malignant (0)', 'Benign (1)'])
            ax.set_xlabel("Tahmin Edilen Sınıf")
            ax.set_ylabel("Gerçek Sınıf")
            ax.set_title("TabNet – Confusion Matrix")
            st.pyplot(fig)
        st.write("---")

        # 4) 10-Fold Çapraz Doğrulama Skorları
        cv_scores = [
            1.0, 0.9824561404, 0.9824561404, 0.9473684211, 0.9649122807,
            0.9824561404, 1.0, 0.9824561404, 1.0, 0.9821428571
        ]


        # 5) Ortalama CV Doğruluk
        st.subheader("5) Ortalama CV Doğruluk")
        st.metric(label="Ortalama", value=f"{np.mean(cv_scores):.10%}")

        # Bar Chart
        st.bar_chart(
            pd.DataFrame({"Fold Doğruluk": cv_scores},
                        index=[f"Fold {i}" for i in range(1, 11)])
        )

        st.write("")  # son boşluk

    ## ./TABNET ## 

    st.write('---')

    ## MODELLERIN ORTALAMA ACC KARSILASTIRMA ##
    # Modeller ve doğruluk oranları
    
    #dataframe olustur
    models = ["TabNet", "LSTM", "MLP", "CNN" ]
    accuracies = [ 98.2424812050, 97.7192982000, 97.3684210000, 97.3623188410]

    
    df = pd.DataFrame({
        "Model": models,
        "Ortalama Doğruluk (%)": [f"{acc:.10f}" for acc in accuracies]
    })

    st.header("Derin Öğrenme Modellerinin Ortalama Doğruluk Oranları")

    col1, col2 = st.columns(2)

    with col1:    
        # Başlık ve tabloyu göster
        st.table(df)

    with col2:
        # DataFrame oluştur
        df = pd.DataFrame({
            "Model": models,
            "Accuracy (%)": accuracies
        })

        # Renkli çubuk grafik (Altair)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Model:N", title="Model"),
                y=alt.Y("Accuracy (%):Q", title="Accuracy (%)"),
                color=alt.Color("Model:N", legend=None)
            )
            .properties(
                width="container",
                height=400
            )
        )

        # Grafiği göster
        st.altair_chart(chart, use_container_width=True)

    st.header("Sonuç")
    boxInfo("""
    Bu çalışma kapsamında, **Breast Cancer Wisconsin (Diagnostic) Data Set** kullanılarak dört farklı derin öğrenme modeli eğitilmiş ve 10-katlı çapraz doğrulama ile test edilmiştir.

    - **TabNet**, tüm modeller arasında **en yüksek doğruluğu** (%98.24) elde ederek göze çarpmaktadır. Özellikle **tabular veri** üzerindeki başarımı ile dikkat çeken bu model, veri içindeki ilişkileri verimli biçimde modelleyebilmiştir.
    - **LSTM**, zaman serisi temelli yapısına rağmen bu sabit veri setinde güçlü bir genel başarı (%97.71) sergilemiştir. Özellikle örüntü tanımadaki avantajı, sınıflar arası ayırt ediciliği arttırmış olabilir.
    - **MLP ve CNN**, birbirine oldukça yakın doğruluk değerlerine sahiptir (%97.36 civarında). Bu durum, her iki modelin de temel düzeyde nöral ağ yapısıyla bu tür sabit özellikli veri setinde yeterince başarılı olduğunu göstermektedir.
    - **CNN**’in doğruluğu, LSTM ve TabNet’e kıyasla biraz daha düşük kalmıştır. Bunun temel nedeni, CNN’nin genellikle uzamsal ilişkiler için optimize edilmiş olmasıdır; bu veri setinde ise özellikler arasında böyle bir uzamsal yapı bulunmamaktadır.
    ---
    Özetle **En yüksek genel doğruluk TabNet modelinde elde edilmiştir**, bu nedenle breast cancer teşhisi gibi hayati önem taşıyan sınıflandırma görevlerinde, tabular veriler için özel olarak geliştirilmiş bu mimarinin tercih edilmesi uygun olacaktır. **LSTM ve MLP gibi klasik sinir ağı yaklaşımları da oldukça güçlü alternatiflerdir**, özellikle model karmaşıklığının azaltılması veya yorumlanabilirlik arandığında değerlendirilebilir. Sonuçlar, veri setinin özellik mühendisliği açısından zaten güçlü olduğunu ve bu nedenle temel modellerin bile oldukça yüksek başarı sağlayabildiğini göstermektedir.
    """, "green")

   
    ## ./MODELLERIN ORTALAMA ACC KARSILASTIRMA ##

    st.write('---')

    ## HEM GELENEKSEL HEM DE DERIN OGRENME MODELLERIN KARSILATIRILMASI ##
    st.title("Tüm Modellerin Değerlendirilmesi")

    models = [
    "TabNet", "SVM", "Logistic Regression", "LSTM",
    "MLP", "CNN", "KNN", "XGBoost", "Random Forest"
    ]
    accuracies = [
        98.2425, 97.8885, 97.7971, 97.7193,
        97.3684, 97.3623, 97.0113, 96.0290, 95.5990
    ]

    # DataFrame oluştur: modelleri ve karşılık gelen doğrulukları eşleştir
    df = pd.DataFrame({
        "Model": models,
        "Ortalama Doğruluk (%)": accuracies
    })

    # Tabloyu göster: doğruluğa göre büyükten küçüğe sırala
    df_sorted = df.sort_values(by="Ortalama Doğruluk (%)", ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.table(df_sorted)  # Sıralı tabloyu Streamlit arayüzünde göster

    with col2:
        chart = (
            alt.Chart(df_sorted)  # Sıralı DataFrame'i grafik veri kaynağı olarak kullan
            .mark_bar()       # Bar (çubuk) grafik türünü seç
            .encode(
                x=alt.X("Model:N", sort=None, title="Model"),                             # X eksenine model isimleri
                y=alt.Y("Ortalama Doğruluk (%):Q", title="Ortalama Doğruluk (%)"),        # Y eksenine doğruluk yüzdeleri
                color=alt.Color("Model:N", legend=None)                                   # Her çubuğu ayrı renkte göster, legend gizli
            )
            .properties(
                width="container",   # Grafik genişliğini konteynıra uydur
                height=400,          # Grafik yüksekliğini 400 piksel yap
            )
        )

        st.altair_chart(chart, use_container_width=True)  # Grafiği Streamlit'te göster ve genişliği kapsayıncıya uydur

    boxExp("""
    Bu çalışmada, **Breast Cancer Wisconsin (Diagnostic) Data Set** kullanılarak hem geleneksel makine öğrenmesi hem de derin öğrenme modelleri eğitilmiş ve 10-katlı çapraz doğrulama yöntemiyle doğruluk başarımları değerlendirilmiştir.

    ---

    #### Genel Değerlendirme

    - **TabNet** modeli, tüm modeller arasında en yüksek ortalama doğruluk oranını (%98.24) elde ederek öne çıkmaktadır. Bu başarı, özellikle tabular veri ile doğrudan çalışabilen mimarisinden kaynaklanmaktadır. Modelin, veri içerisindeki ilişki örüntülerini etkili şekilde yakalayabildiği görülmektedir.
    
    - **SVM (%97.88)** ve **Logistic Regression (%97.79)** modelleri, geleneksel yöntemler içinde en yüksek başarıyı göstermiştir. Bu modeller, veri setinin düzgün ayrılabilirlik taşıyan yapısından etkili şekilde faydalanabilmektedir.

    - **LSTM (%97.72)** ve **MLP (%97.37)** gibi derin öğrenme modelleri, genellikle sekans ve soyut temsillerle daha başarılı olurken bu sabit tabular veri seti üzerinde de güçlü bir performans ortaya koymuştur. LSTM’nin bu veri setinde başarılı olması, modelin genel genelleme kabiliyetinin yüksekliğini göstermektedir.

    - **Random Forest (%95.60)** ve **XGBoost (%96.03)**, sabit ve sınırlı derinlikteki karar ağaçlarıyla karmaşık örüntüleri yeterince yakalayamayarak, yüksek ayırt ediciliğe sahip veri setlerinde bilgi kaybına ve dolayısıyla düşük genelleme performansına neden olduğu düşünülmüştür.

    - **CNN (%97.36)** modeli genellikle görüntü verilerinde öne çıksa da, bu çalışmada öz nitelikler arasında sınırlı ilişkilere rağmen güçlü bir performans göstermiştir. Bu, doğru ön işlem ve veri temsiliyle CNN'nin tabular verilerde de etkili olabileceğini göstermektedir.

    ---

    #### Sonuç ve Öneriler

    - **Uygulama hassasiyeti yüksek olan tıbbi teşhis sistemlerinde**, doğruluğun yanında modelin yorumlanabilirliği, eğitim süresi ve veri tipiyle uyumu gibi kriterler de göz önünde bulundurulmalıdır.
    - **TabNet**, bu bağlamda en yüksek doğruluğu sağlaması nedeniyle güçlü bir adaydır. Ancak, **Logistic Regression** gibi daha sade ve yorumlanabilir modellerin de başarımı oldukça yüksektir ve operasyonel ortamda tercih edilebilir.
    ---
    """, "green")

    ## ./HEM GELENEKSEL HEM DE DERIN OGRENME MODELLERIN KARSILATIRILMASI ##
