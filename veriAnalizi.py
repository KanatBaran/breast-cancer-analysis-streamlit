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

### AYARLAR (START) ##
# Ortak veri yükleme
@st.cache_data
def load_data(path):
    return pd.read_csv(path)
data = load_data("data.csv")
### AYARLAR (END) ###

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
## ./tasarim foknsiyonlari ##
### FONKSIYONLAR (END) ###

def app():

    # Ana içerik
    st.title("Veri Analizi & Ön İşleme")

    
    boxExp("""
    **Veri Analizi**, verilerin içindeki anlamlı desenleri, eğilimleri ve ilişkileri keşfetmek için yapılan sistematik incelemedir. Amaç, karar verme süreçlerini desteklemek için veriden bilgi çıkarmaktır.

    **Ön İşleme** ise, verilerin analiz veya modelleme için uygun hale getirilmesi sürecidir. Eksik verilerin doldurulması, aykırı değerlerin temizlenmesi, verinin standardize edilmesi gibi adımları içerir.

    Bu adımlar, bir makine öğrenmesi modelinin başarısını doğrudan etkileyen temel hazırlık sürecidir.
    """, "black")
    
    st.header("İlk Bakış")
    col1, col2 = st.columns(2) # 2 adet sutun olsuturmak icin
    
    with col1:
        st.subheader("Sütun İsimleri")
        
        df_columns = pd.DataFrame(data.columns, columns=["Sütun Adı"]) # Orijinal veri kümesinin sütun adlarını yeni bir DataFrame’e aktarır
        df_columns.index = df_columns.index + 1 # İndeks numaralarını 1’den başlatır (varsayılan 0’dan 1’e)
        st.dataframe(df_columns, use_container_width=True) # Sütun adları tablosunu geniş konteyner içinde gösterir

    with col2:
        st.subheader("İlk 10 Kayıt")  # Bölüm başlığı
        st.dataframe(data.head(10), use_container_width=True)  # data.head(10) çıktısını gösterir

        st.write(f"Satır sayısı: {data.shape[0]}, Sütun sayısı: {data.shape[1]}")

    st.write('---')

    ## Eksik Veri Adimi ##
    st.header("Eksik Veri Kontrolü")

    boxInfo("""
    **Eksik Veri (Missing Data)**: Veri setinde bazı hücrelerin boş kalması durumudur.

    **Neden Kaldırılır?**  
    Eksik veriler, analiz ve model performansını olumsuz etkileyebilir. Bu yüzden çoğu zaman ya silinir ya da uygun değerlerle doldurulur.
    """)
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("İşlem Öncesi Sütun Listesi")
        st.write(data.isnull().sum().sort_values(ascending=False))  # data.isnull().sum() çıktısını gösterir.
        # .sort_values(ascending=False) ile buyukten kucuge bir siralama yapildi.

        # Model için anlamsız olan 'id' sütununu kaldırır
        data.drop("id", axis=1, inplace=True)  # 'id' sütunu kaldırıldı

        # Tüm değerleri eksik olan 'Unnamed: 32' sütununu kaldırır
        data.drop("Unnamed: 32", axis=1, inplace=True)  # 'Unnamed: 32' sütunu kaldırıldı

    with col2:
        # Güncel sütun adlarını göstermek istersen
        st.subheader("İşlem Sonrası Sütun Listesi")
        st.dataframe(pd.DataFrame(data.columns, columns=["Sütun Adı"]), use_container_width=True)

        
    boxInfo("""
    **Ne Yaptım?** 
    <br>Veri setini Kaggle.com üzerinden indirdiğimde, '***Unnamed: 32***' adında tamamen boş bir sütun ve model eğitimi için anlamlı bir bilgi taşımayan '***id***' adlı bir sütun olduğunu fark ettim. Eksik veri analizi sırasında bu sütunları tespit ederek veri setinden çıkardım.

    """, "green")


    
    ## ./Eksik Veri Adimi ##

    st.write('---')

    ## Genel Analiz Adimi ##
    st.header("Genel Analiz Adımı")

    boxInfo("""
    Bu yapıları, veri setini modelleme öncesinde daha iyi anlayabilmek için oluşturdum.

    - **Diagnosis dağılım grafiği** ile hedef değişkenin sınıf dengesini görselleştirdim.  
    - **Tanımlayıcı istatistikler tablosu** ile sayısal değişkenlerin temel istatistiksel özelliklerini inceledim.

    Bu sayede verinin genel yapısını özetleyerek sonraki adımlara hazır hale getirdim.
    """)


    col1, col2 = st.columns(2)

    with col1:
        # figür boyutunu 6x4 inç olarak ayarladık
        fig, ax = plt.subplots(figsize=(3, 2))

        st.subheader("Hedef Değişkenler")
        sns.countplot(
            x="diagnosis",
            hue="diagnosis",
            data=data,
            palette={"M": "red", "B": "green"},
            ax=ax
        )  # renk ayrımı: M için kırmızı, B için yeşil

        ax.set_title("Diagnosis Dağılımı")   # grafik başlığı
        ax.set_xlabel("Teşhis Türü")         # X ekseni etiketi
        ax.set_ylabel("Sayı")                # Y ekseni etiketi
        st.pyplot(fig, use_container_width=False)

    with col2:
        # Tanımlayıcı İstatistikler: sayısal sütunlar için ortalama, std, min, max vb. bilgileri gösterir
        st.subheader("Tanımlayıcı İstatistikler")
        desc = data.describe()                  # describe() ile temel istatistiksel bilgileri alır
        st.dataframe(desc, use_container_width=True)  # istatistikleri tablo halinde gösterir
    ## ./Genel Analiz Adimi ##

    st.write('---')

    ## Ortlama ve Standart Sapma Kontrol Adimi ##
    st.header("Ortalama ve Standart Sapma Kontrol Adımı")

    boxInfo("""
    **Ortalama (Mean)**: Verilerin genel merkezini temsil eder. Tüm değerlerin toplamının, veri sayısına bölünmesiyle elde edilir.  

    **Standart Sapma (Std)**: Verilerin ortalama etrafında ne kadar yayıldığını gösterir. Yani, değerlerin ortalamadan ne kadar sapma gösterdiğini ifade eder.

    Bu değerlere bakarak veri setindeki dağılımı gözlemliyorum. Böylece değişkenler arasında ölçek farkı olup olmadığını anlayıp uygun bir ***ölçeklendirme yöntemi*** belirleyeceğim.
    """)
    st.write("")

    st.subheader("Ortalama Kontrolü")

    col1, col2 = st.columns(2)
    with col1:
        # Tanımlayıcı istatistiklerden ortalamaları alıyoruz ve büyükten küçüğe sıralıyoruz
        means = desc.loc["mean"].sort_values(ascending=False)  # tüm sütunların ortalama değerlerini alır

        # Ortalama değerleri tablo halinde gösteriyoruz
        st.dataframe(means.to_frame(name="Ortalama Değer"), use_container_width=True)  # means serisini DataFrame'e dönüştürüp gösterir

    with col2:
        # Çubuk grafik ile ortalamaları görselleştiriyoruz
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)     # figür boyutunu ve dpi'ı ayarlar
        means.plot(kind="bar", ax=ax)                       # bar plot çizer
        ax.set_title("Özelliklerin Ortalama Değerleri")     # grafik başlığı
        ax.set_ylabel("Ortalama Değer")                     # Y ekseni etiketi
        ax.set_xlabel("")                                   # X ekseni etiketi (boş bırakılabilir)
        ax.tick_params(axis="x", rotation=90)               # X etiketlerini dikeyden yataya döndürür
        ax.grid(axis="y")                                   # Y eksenine grid ekler
        plt.tight_layout()                                  # düzeni sıkıştırarak taşmaları engeller

        # Grafiği Streamlit ana alanına ekler; container genişliğini kullanmaz
        st.pyplot(fig, use_container_width=False)

    # Standart Sapma
    st.subheader("Standart Sapma Kontrolü")

    col1, col2 = st.columns(2)

    with col1:
        # Sadece sayısal sütunları al (diagnosis gibi kategorik sütunları çıkar)
        numeric_df = data.select_dtypes(include="number")
        
        # Her sütunun standart sapmasını hesapla
        std_table = numeric_df.std().sort_values(ascending=False).to_frame("Standart Sapma")
        std_table.index.name = "Özellik"
        
        # Streamlit’te tabloyu göster
        st.dataframe(std_table)

    with col2:
        # Grafik oluşturma
        fig, ax = plt.subplots(figsize=(8, 6))
        desc.loc["std"].sort_values().plot(kind="barh", ax=ax)
        ax.set_title("Özelliklerin Standart Sapmaları")
        ax.set_xlabel("Standart Sapma")
        plt.tight_layout()

        # Grafiği Streamlit’e ekle
        st.pyplot(fig, use_container_width=False)


    boxInfo("""
    Ortalama ve standart sapma değerlerine baktığımda, bazı özelliklerin (özellikle **area_worst** ve **area_mean**) diğerlerine kıyasla çok daha yüksek değerlere ve varyansa sahip olduğunu fark ettim.  
    Bu durum, model eğitiminde bu özelliklerin aşırı ağırlık kazanmasına ve dengesiz kararlara yol açabileceğini gösteriyor.  
    Bu nedenle tüm özellikleri ortak bir ölçeğe çekmek için ***StandardScaler*** ile standartlaştırma yapmaya karar verdim.
    """, "green")

    st.write("")
    st.write("")
    
# Özellik Ölçeklendirme Adımı
    st.subheader("Özellik Ölçeklendirme Adımı (StandartScaler)")
    boxInfo("""
    **StandardScaler Nedir?**  
    Scikit-learn kütüphanesinin bir ön işleme aracıdır.

    **Nasıl Çalışır?**  
    - Eğitim verisindeki her özelliğin ortalaması (μ) hesaplanır.  
    - Her değerden bu ortalama çıkartılır: 𝑥′ = 𝑥 − μ  
    - Ortalamadan arındırılan değer, ilgili özelliğin standart sapmasına (σ) bölünür: 𝑥″ = 𝑥′ / σ  
    - Sonuçta tüm özellikler ortalama=0 ve standart sapma=1 olacak şekilde dönüştürülür.  

    **Kaynakça:**  
    Scikit-learn Developers. (2024). *StandardScaler*. Scikit-learn documentation. <br>
    Erişim: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    """)

    st.write("")

    # Ölçeklendirme öncesi ve sonrası tablolarını yan yana göstermek için iki sütun oluşturuyoruz
    oncesi, sonrasi = st.columns(2)

    with oncesi:
        st.subheader("**Ölçeklendirme Öncesi (ilk 5 satır)**")  # bölüm başlığı
        st.dataframe(
            data.drop("diagnosis", axis=1).head(5),              # teşhis sütunu hariç orijinal veri
            use_container_width=True
        )

    # Scaler nesnesi oluşturulur ve uygulanır
    scaler = StandardScaler()                                   # ortalama=0, std=1 ölçeklendirme yapacak
    scaled_values = scaler.fit_transform(data.drop("diagnosis", axis=1))  
    data_scaled = pd.DataFrame(                                  # ölçeklenmiş DataFrame oluşturma
        scaled_values,
        columns=data.columns.drop("diagnosis")
    )
    data_scaled["diagnosis"] = data["diagnosis"].values         # hedef sütunu geri ekleme

    with sonrasi:
        st.subheader("**Ölçeklendirme Sonrası (ilk 5 satır)**")  # bölüm başlığı
        st.dataframe(
            data_scaled.drop("diagnosis", axis=1).head(5),      # ölçeklenmiş veri
            use_container_width=True
        )


    boxInfo("""
    **StandardScaler’i neden uyguladım?**  
    Verideki özellikler çok farklı ölçeklere sahipti (örneğin `area_mean` binlerce, `smoothness_se` ise küçük ondalık değerler).  
    Bu ölçek farkı, modelin bazı özelliklere aşırı ağırlık vermesine yol açabilirdi.  
    Bu yüzden tüm özellikleri ortalama=0, sapma=1 olacak şekilde standartlaştırdım. Böylece:
    - **Özelliklerin eşit katkısını** sağladım.  
    - **Yakınsama süresini** hızlandırarak eğitim sürecini kararlı kıldım.  
    - SVM, KNN gibi **mesafeye dayalı algoritmaların** performansını artırdım.  
    - Aşırı büyük değerlerin yol açabileceği **sayısal kararsızlıkları** önledim.  
    - Modelimin **genelleştirme yeteneğini** güçlendirdim.
    """, "green")
        ## ./Ortlama ve Standart Sapma Kontrol Adimi ##

    st.write('---')
    
    ## Aykırı Deger Adimi ##
    st.header("Aykırı Değer Adımı")

    boxInfo("""
    **Aykırı Değerlere Boxplot & IQR ile Bakma Yöntemim**  
    Verideki uç değerleri görsel olarak tespit etmek için boxplot grafiği kullanacağım.  
    Boxplot’ta kutunun alt çeyreği (Q1) ile üst çeyreği (Q3) arasındaki mesafe, yani **IQR (Çeyrekler Arası Aralık)** üzerinden çalışıyorum.  

    - **IQR Hesabı:**  
    IQR = Q₃ − Q₁  

    - **Aykırı Değer Kriteri:**  
    • Alt sınır = Q₁ − 1.5 × IQR  
    • Üst sınır = Q₃ + 1.5 × IQR  

    Bu sınırların dışında kalan gözlemleri potansiyel aykırı değer olarak işaretleyip inceleyeceğim.

    **Kaynakça:**  
    Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.   
    """)

    # Boxplot figürü oluşturulur (ölçeklenmiş veri, 'diagnosis' hariç)
    fig, ax = plt.subplots(figsize=(8, 4))  # figür boyutu: 8x4 inç
    sns.boxplot(
        data=data_scaled.drop("diagnosis", axis=1),  # hedef sütunu çıkar
        orient="h",                                   # yatay orientasyon
        ax=ax
    )
    ax.set_title("Aykırı Değerlerin Boxplot ile Gösterimi (Scaled Veri)")  # grafik başlığı
    plt.tight_layout()  # düzen sıkıştırma

    # Grafiği Streamlit ana alanına ekle; konteyner genişliğini kullanma
    st.pyplot(fig, use_container_width=False)


    boxInfo("""
    <b>Ne Yaptım?</b><br>
    Veri setinde çok sayıda aykırı değer tespit ettim. Tıp alanındaki veri setlerinde uç değerlerin bulunması sık rastlanan bir durum ve bu değerler klinik çeşitliliği, nadir vakaları yansıtır.  
    Bu aykırı değerlerin model eğitimi için önemli olduğunu düşündüğümden, silmek yerine modelin bu uç örneklerden öğrenmesini sağlayarak daha gerçekçi ve genelleyici sonuçlar elde etmeyi tercih ettim.
    """, "green")
    ## ./Aykırı Deger Adimi ##

    

    st.write('---')

    ## ./Korelasyon Adimi ##
    # Korelasyon Adımı ve Özellik Seçimi
    st.header("Korelasyon Adımı ve Özellik Seçimi")

    boxInfo("""
    **Korelasyon Nedir?**  
    Korelasyon, iki sayısal değişken arasındaki doğrusal ilişkinin yönünü ve gücünü ölçmek demek. Değerler -1 ile 1 arasında değişir; 1 tam pozitif, -1 tam negatif ilişkiyi gösterir.  

    **Neden Korelasyona Bakıyorum?**  
    Veri setimde birbirine çok benzeyen (yüksek korelasyonlu) özellikler modelde redundant bilgiye ve multicollinearity’ye yol açabilir. Hangi değişkenlerin birbirine fazlaca bağlı olduğunu görüp gereksiz olanları elemek için korelasyon analizini kullanıyorum.
    """)

    # Sayısal sütunları seç
    x = data_scaled.select_dtypes(include=['float64', 'int64'])

    # Figür ve eksen oluştur, boyutunu ayarla
    fig, ax = plt.subplots(figsize=(16, 8))

    # Heatmap’i çiz
    sns.heatmap(
        x.corr(),
        annot=True,
        linewidths=0.5,
        fmt='.1f',
        ax=ax
    )
    ax.set_title("Korelasyon Haritası")

    # Streamlit’e doğru figürü ekle
    st.pyplot(fig, use_container_width=True)

    # Sayısal sütunları seç
    x = data_scaled.select_dtypes(include=['float64', 'int64'])

    # Korelasyon matrisini mutlak değer olarak al
    corr_matrix = x.corr().abs()

    # Üst üçgeni (i<j) al
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Korelasyonu 0.95'ten büyük olan çiftleri yakala
    high_pairs = (
        upper
        .stack()                             # Çok seviyeli index serisi: (Feature A, Feature B) -> corr
        .reset_index(name='r')               # DataFrame’e çevir
        .rename(columns={'level_0':'Feature A','level_1':'Feature B'})
    )

    # Sadece r>0.95 çiftleri al, istersen sıralayabilirsin
    high_pairs = high_pairs[high_pairs['r'] > 0.95].sort_values('r', ascending=False)

    # Streamlit’te göster
    st.markdown("### r > 0.95 korelasyona sahip çiftler")
    st.dataframe(high_pairs, use_container_width=True)
    
    boxInfo("""
    <b>Ne Yaptım?</b><br>
    Tabloya göre **r > 0.95** korelasyona sahip özellik çiftlerinden her grupta bir temsilci özellik seçip diğerini kaldıracağım. Ancak bu süreçte çok dikkatli olmam gerekiyor: **0.95** üstü korelasyon bile birçok sütun arasında yaygın. Yanlış bir özellik silimi, modelin doğruluğunu olumsuz etkileyebilir.  
    Bu nedenle her adımı titizlikle inceleyerek, model performansını koruyacak şekilde seçim yapmalıyım.
    """, "green")

    ## ./Korelasyon Adimi ##

    st.write('---')

    ## Hedef Degisken Adimi ##
    st.subheader("Hedef Değişken Adımı")

    # Orijinal (B/M) sınıfları korumak için veri kümesinin bir kopyasını alıyoruz
    data_unmapped = data.copy()

    # Eğer diagnosis hâlâ string ise, B=1, M=0 şeklinde sayısala dönüştürüyoruz
    if data_scaled["diagnosis"].dtype == object:
        data_scaled["diagnosis"] = data_scaled["diagnosis"].map({"B": 1, "M": 0})

    # Kullanıcıya neden bu dönüşümü yaptığımızı açıklıyoruz
    boxInfo("""
    **Neden ‘diagnosis’ Etiketlerini Sayısallaştırdım?**  
    Makine öğrenmesi algoritmalarının yalnızca sayısal girdilerle çalıştığını bildiğim için, sınıflandırma hedefim olan ‘diagnosis’ sütunundaki kategorik etiketleri dönüştürdüm.  
    - “B” (Benign) değerini **1**,  
    - “M” (Malign) değerini **0**  

    olarak kodladım. Bu sayede modelim, hedef değişkeni doğrudan işleyebildi ve performansını artırdım.  
    """)

    # Orijinal ve dönüştürülmüş sınıf dağılımlarını yan yana gösteriyoruz
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1) Orijinal (B/M) dağılım
    sns.countplot(
        x="diagnosis",
        hue="diagnosis",
        data=data_unmapped,
        ax=axes[0],
        palette="Set2",
        legend=False
    )
    axes[0].set_title("Orijinal Sınıflar (B/M)")
    axes[0].set_xlabel("Sınıf")
    axes[0].set_ylabel("Adet")

    # 2) Dönüştürülmüş (1/0) dağılım
    sns.countplot(
        x="diagnosis",
        hue="diagnosis",
        data=data_scaled,
        ax=axes[1],
        palette="Set1",
        legend=False
    )
    axes[1].set_title("Dönüştürülmüş Sınıflar (B=1, M=0)")
    axes[1].set_xlabel("Sınıf")
    axes[1].set_ylabel("Adet")

    plt.tight_layout()

    # Grafiği Streamlit ana alanına ekliyoruz
    st.pyplot(fig, use_container_width=True)

    ## ./Hedef Degisken Adimi ##
    st.write("---")
    st.header("Sonuç")
    boxInfo("""
    Veri analizi ve ön işleme adımlarını tamamladım:  
    - **Eksik veri** kontrolü ve gereksiz sütunların temizlenmesi  
    - **Ortalama** ve **standart sapma** analizleri  
    - **Aykırı değer** tespiti ve uygun yaklaşımla korunması  
    - **Korelasyon** incelemesi ve yüksek ilişkili özelliklerin elenmesi  

    Artık elde ettiğim bu temiz ve dengelenmiş veri setiyle, hem geleneksel makine öğrenme modelleri hem de derin öğrenme modelleri üzerinde test ve değerlendirmelerime geçeceğim.
    """, "green")

    # En son elde ettigimiz veri setini kaydetmemiz gerekiyor.
    st.session_state["data_scaled"] = data_scaled


        