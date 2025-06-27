### KUTUPHANELER (START) ###
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
import seaborn as sns
### KUTUPAHANELER (END) ###


### FONKSİYONLAR (START) ###

# CSV dosyasını okuyup DataFrame olarak döndüren fonksiyon
def dataLoad(path="data.csv"):
    return pd.read_csv(path)


# Eksik Veri Adimi #
# Veri setindeki eksik verileri kontrol ediyor. Varsa eksik veriye ait sütunu döndürüyor.
def missingValue(df):
    st.subheader("Veri Setinde Eksik Değer Kontrolü")
    missing_counts = df.isnull().sum()
    # Sadece eksik değeri olan sütunları göster
    st.write(missing_counts[missing_counts > 0].to_frame("Eksik Değer Sayısı"))


# Satır bazında eksik satırları siler.
def dropMissingRows(df):
    return df.dropna(axis=0)


# Sütun bazında eksik sütunları siler.
def dropMissingCols(df):
    return df.dropna(axis=1)
# ./Eksik Veri Adimi #


# Aykiri Deger Adimi #
# Veri setindeki aykiri degerleri 'Box Plot Grafiği' ile gosterir.
def plotBoxplots(df):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    st.subheader("Aykırı Değer Kontrolü (Boxplot)")
    for col in numeric_cols:
        # Her bir sayısal sütun için ayrı figür oluştur
        fig, ax = plt.subplots()
        ax.boxplot(df[col].dropna())
        ax.set_title(f"{col} için Boxplot")
        st.pyplot(fig)
    st.write("---")


# IQR yontemiyle sayisal sutunlarda aykiri degerleri tespit eder; IQR disindaki degerler aykiri sayilir.
def detectOutliersIqr(df):
    numeric = df.select_dtypes(include=np.number)
    Q1 = numeric.quantile(0.25)
    Q3 = numeric.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Her sütun için aykırı değer maskesi
    mask = (numeric < lower_bound) | (numeric > upper_bound)
    # Satır bazlı aykırı kontrol: herhangi bir sütunda True ise o satır aykırı
    return mask.any(axis=1)


# Aykiri deger olarak isaretlenen satirlari siler.
def removeOutliers(df):
    mask = detectOutliersIqr(df)
    return df.loc[~mask]
# ./Aykiri Deger Adimi #
### FONKSİYONLAR (END) ###


### DEGISKENLER (START) ###
# Yanlis donusum secimi icin kullanilmaktadir
if "df_backup" not in st.session_state:
    st.session_state.df_backup = None
### DEGISKENLER (END) ###


### BASLANGIC (START) ###
def app():
    st.title("Veri Analizi & Ön İşleme")

    st.markdown("""
        **Veri Analizi ve Ön İşleme Nedir?**

        Veri Analizi, ham verinin yapısını anlamak, eksik veya hatalı noktaları tespit etmek ve genel eğilimleri keşfetmek amacıyla veriye göz atma sürecidir. Ön İşleme ise bu analiz sonuçlarına dayanarak veriyi makine öğrenmesi modelleri için hazır hâle getirme adımlarını kapsar.

        **Neden Önemlidir?**
        - **Doğru Kararlar İçin Temel:** Ham verideki eksik değerler, tutarsızlıklar veya aykırı (outlier) noktalar model performansını ciddi şekilde düşürebilir. Ön işleme sayesinde bu sorunları gidererek daha güvenilir sonuçlar elde ederiz.
        - **Model Başarımını Artırır:** Veriyi ölçekleme, eksik değerleri doldurma ya da temizleme, aykırı değerleri ele alma gibi adımlar, algoritmanın öğrenme sürecini iyileştirir; böylece tahmin doğruluğu yükselir.
        - **Zaman ve Kaynak Tasarrufu:** Temizlenmemiş veriler üzerinde çalışmak uzun hesaplama sürelerine ve gereksiz hatalara yol açar. Ön işleme aşaması, gereksiz kayıtları elemek ve veri kalitesini yükseltmek için zaman kazandırır.
        - **Genelleştirme Kabiliyetini Güçlendirir:** Düzenli ve tutarlı bir veri seti, modelin daha önce görmediği yeni veriler üzerinde de iyi performans göstermesini sağlar.

        Bu uygulamada öncelikle verinin kalitesi; eksik değerler, aykırı gözlemler, dağılım, standart sapma gibi istatistiksel ölçütlerle değerlendirilip temizlenecektir. Ardından işlenmiş hâle getirilen veri dışa aktarılarak “Model Eğitimi” sayfasında makine öğrenmesi adımlarına geçilecektir.
    """)

    st.write("---")


    ## VERI SETI YUKLEME ##
    st.subheader("1. VERİ YÜKLEME")
    st.markdown(
        "**Varsayılan Veri Seti:** Breast Cancer Wisconsin (Diagnostic) Data Set. "
        "[İncele](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)"
    )
    st.write(
        "Lütfen analizinizde kullanmak üzere uygun bir CSV dosyasını yükleyiniz; "
        "herhangi bir dosya yüklemezseniz otomatik olarak belirtilen varsayılan veri seti kullanılacaktır."
    )

    uploaded_file = st.file_uploader("CSV dosyanızı yükleyin", type=["csv"])

    if "df_original" not in st.session_state:
        # İlk yükleme anında ya yüklenen dosyayı ya da varsayılanı al
        if uploaded_file is not None:
            try:
                st.session_state.df_original = pd.read_csv(uploaded_file)
                st.success("Yüklenen dosya başarıyla okundu ve session_state'e kaydedildi.")
            except Exception as e:
                st.error(f"Yüklenen dosya okunurken hata oluştu: {e}")
                st.stop()
        else:
            try:
                st.session_state.df_original = dataLoad("data.csv")
                st.info("Varsayılan veri seti kullanılıyor: `data.csv - Breast Cancer Wisconsin (Diagnostic) Data Set`")
            except FileNotFoundError:
                st.error("Varsayılan `data.csv` bulunamadı. Lütfen kendi CSV dosyanızı yükleyin.")
                st.stop()

    df_original = st.session_state.df_original.copy()

    ## 1.1 VERİ SETİ DÜZENLEME ##
    # Yüklü veri setini tablo olarak göster
    st.markdown("**Yüklü Veri Seti Önizleme**")
    st.dataframe(df_original)

    st.subheader("1.1 Veri Seti Düzenleme")

    # --- Yan yana seçim kutuları ---
    col1, col2 = st.columns(2)
    with col1:
        to_drop_cols = st.multiselect(
            "Silinecek sütunlar",
            options=st.session_state.df_original.columns.tolist()
        )
    with col2:
        to_drop_rows = st.multiselect(
            "Silinecek satır indeksleri",
            options=st.session_state.df_original.index.tolist(),
            format_func=lambda x: str(x)
        )

    # --- Tek butonla silme ---
    if st.button("Seçilen Satır ve Sütunları Sil"):
        if not to_drop_cols and not to_drop_rows:
            st.info("Lütfen en az bir sütun veya satır seçin.")
        else:
            if to_drop_cols:
                st.session_state.df_original.drop(columns=to_drop_cols, inplace=True)
                st.success(f"Sütunlar silindi: {', '.join(to_drop_cols)}")
            if to_drop_rows:
                st.session_state.df_original.drop(index=to_drop_rows, inplace=True)
                st.success(f"Satırlar silindi: {to_drop_rows}")
            # Silme sonrası güncellenmiş veri setini göster
            st.markdown("**Güncellenmiş Veri Seti Önizleme**")
            st.dataframe(st.session_state.df_original)

        ## ./VERI SETI YUKLEME ##

    st.write("---")


    ## EKSIK VERI ADIMI ##
    st.subheader("2. EKSİK VERİ")

    with st.expander("Eksik Veri Nedir ve Neden Silinmelidir?"):
        st.markdown("""
        **Eksik Veri Nedir?**

        Eksik veri (missing data), bir veri setindeki bazı gözlemlerde belirli değişkenlere ilişkin değerlerin kaybolması veya ölçülmemesi durumu olarak tanımlanır. Eksik değerler çeşitli nedenlerle ortaya çıkabilir: Ölçüm hataları, veri toplama sürecindeki aksaklıklar, katılımcıların soruya yanıt vermemesi veya veri kayıt sistemlerinin sınırlamaları gibi etkenler bu duruma yol açar. Eksik veriler örnek üç kategoriye ayrılabilir (Little & Rubin, 2002):
        1. **MCAR (Missing Completely At Random):** Eksik değer oluşumu, ne tamamlama ne de gözlemlenen diğer değişkenlerle ilişkili değildir.
        2. **MAR (Missing At Random):** Eksiklik, eksik olmayan başka bir değişken tarafından açıklanabilir, ancak eksikliğin kendisi doğrudan diğer gözlemlenen değişkenlere bağlı değildir.
        3. **MNAR (Missing Not At Random):** Eksiklik doğrudan gözlemlenemeyen bir mekanizma ile ilişkilidir (ör. katılımcı sağlığı kötü hissettiği için yanıt vermemiş olabilir).

        **Neden Silinmelidir?**

        1. **Model Performansını Artırmak İçin:**  
        Eksik gözlemler, özellikle birçok makine öğrenmesi algoritmasının eksik veri içerdiğinde doğru çalışmamasına yol açabilir. Silme işlemi, geriye yalnızca tam ve tutarlı kayıtları bırakarak modelin daha güvenilir öğrenmesine imkân sağlar.

        2. **Aşırı Sapmayı (Bias) Önlemek:**  
        Bazı durumlarda, eksik değerler rastgele değil de belirli bir mekanizmayla ortaya çıkar (MAR veya MNAR). Bu tür eksik veriler, belirli alt gruplardan (örneğin hastalık düzeyi yüksek hasta grupları) eksik kayıt içeriyorsa, hem ortalama değerlerin hem de model çıktılarının sistematik olarak sapmasına neden olabilir. Sorunlu gözlemleri tüm satır veya sütun bazında silmek, bu yanlılığı (bias) azaltabilir.

        3. **Analiz Kolaylığı Sağlamak:**  
        Eksik veri içeren kayıtlar, istatistiksel analizlerde ve görselleştirmelerde ek karmaşıklık yaratır. Satır bazlı silme veya sütun bazlı silme, veri setindeki eksik değer problemini ortadan kaldırır ve sonraki adımlarda kodun daha basit ve hata riski düşük çalışmasını sağlar.

        4. **Aşırı Karmaşıklıktan Kaçınmak:**  
        İleri seviye eksik veri yöntemleri (örneğin multiple imputation, regresyon tahmini vb.) kullanmak mümkündür ancak çoğu durumda özellikle eğitim aşamasında zamandan ve hesaplama gücünden tasarruf etmek adına, öncelikle eksik veriyi silme seçeneği değerlendirilebilir. Silme işlemi basit bir adım olduğu için projenin başlangıç aşamalarında hızlı geri dönüş sağlar.

        Bu nedenlerle, veri setindeki eksik kayıtların ilk olarak tespit edilip, duruma göre satır veya sütun bazında silinmesi, model başarımı ve analiz doğruluğu açısından kritik bir ön işleme adımıdır.

        — Little & Rubin (2002), *Statistical Analysis with Missing Data*
        """)

    missingValue(st.session_state.df_original)

    removal_option = st.radio(
        "Eksik veriler için işlem seçin:",
        ("Hiçbirini Silme", "Satır Bazında Sil", "Sütun Bazında Sil"),
        index=0
    )

    if st.button("Eksik Verileri Sil"):
        if removal_option == "Satır Bazında Sil":
            st.session_state.df_original = dropMissingRows(st.session_state.df_original)
            st.success(
                f"Satır bazında eksik değerler silindi. "
                f"Şu anki satır sayısı: {st.session_state.df_original.shape[0]}"
            )
        elif removal_option == "Sütun Bazında Sil":
            st.session_state.df_original = dropMissingCols(st.session_state.df_original)
            st.success(
                f"Sütun bazında eksik değerler silindi. "
                f"Şu anki sütun sayısı: {st.session_state.df_original.shape[1]}"
            )
        else:
            st.info("Eksik veriler korunuyor.")
    else:
        st.info("Eksik veriler için işlem seçilmedi veya silme butonuna basılmadı.")

    # Güncel veri setini göster
    st.subheader("Güncel Veri Seti Önizlemesi")
    st.dataframe(st.session_state.df_original)
    ## ./EKSIK VERI ADIMI ##

    st.write("---")

    ## AYKIRI DEGER ADIMI ##
    numeric_cols = st.session_state.df_original.select_dtypes(include=np.number).columns.tolist()

    st.subheader("3. AYKIRI DEĞER")
    with st.expander("Aykırı Veri Nedir ve Neden İşlenmelidir?"):
        st.markdown("""
        **Aykırı Veri Nedir?**

        Aykırı veri (outlier), bir veri setindeki genel dağılımın dışında kalan, diğer gözlem noktalarından belirgin şekilde uzak değerlerdir. Bu değerler; ölçüm hataları, veri girişindeki yanlışlıklar veya veri toplama sürecindeki anormallikler sebebiyle ortaya çıkabilir. Ayrıca, gerçek anlamda farklı davranış sergileyen uç gözlemler olarak da kabul edilebilir. Aykırı veriler genellikle üç kaynaktan meydana gelir:
        1. **Ölçüm Hataları:** Sensör veya insan kaynaklı kayıt hatalarından doğan tutarsız kayıtlar.
        2. **Veri Giriş Yanlışlıkları:** Elle veri girme sırasında oluşan yazım veya aktarım hataları.
        3. **Gerçek Uç Noktalar:** Gerçekten dağılımın dışında kalan, farklı bir alt gruba ait olabilecek aşırı değerler.

        **Neden İşlenmelidir?**

        1. **Model Performansını Artırmak İçin:**  
        Aykırı değerler, özellikle lineer regresyon, k-en yakın komşu veya mesafe tabanlı yöntemlerde modelin dengesiz öğrenmesine neden olabilir. Aykırı ancak gerçek olmayan kayıtlar kaldırıldığında veya düzeltilme uygulanınca model daha tutarlı sonuçlar üretir.

        2. **Parametrik İstatistiklerin Doğruluğunu Korumak:**  
        Ortalama, varyans gibi parametrik istatistikler, uç değerlerden fazlaca etkilenir. Bu değerler temizlenmeden analiz yapmak, merkezi eğilim ve dağılım ölçülerini çarpıtabilir, hatalı yorumlara sebep olabilir.

        3. **Yüksek Varyansın Azaltılması:**  
        Aykırı gözlemler, modelde yüksek sapma (variance) yaratıp aşırı öğrenmeye (overfitting) yol açabilir. Aykırı değerleri çıkarmak veya sınırlamak, modelin genelleme kabiliyetini güçlendirir.

        4. **Analiz ve Görselleştirme Kolaylığı:**  
        Boxplot, histogram veya scatter plot gibi grafiklerde uç noktalar ölçeği bozarak veri görselleştirmesini zorlaştırır. Bu değerlerin uygun şekilde ayıklanması, dağılımın tamamını daha anlaşılır hâle getirir.

        Bu nedenlerle, aykırı değerlerin tespit edilip duruma göre çıkarılması veya dönüştürülmesi, hem ön işleme sürecinin hem de makine öğrenmesi adımlarının başarısını olumlu yönde etkiler.

        > Duarte, M. M. G., & Sakr, M. (2024). *An experimental study of existing tools for outlier detection and cleaning in trajectories.* GeoInformatica.
        """)

    if numeric_cols:
        seçenekler = ["Tümü"] + numeric_cols
        seçilen_sütun = st.selectbox("Boxplot için sütun seçin:", seçenekler)

        if seçilen_sütun == "Tümü":
            mask_all = detectOutliersIqr(st.session_state.df_original)
            total_outliers = mask_all.sum()
            st.write(f"**Toplam aykırı satır sayısı:** {total_outliers}")

            # Tüm sayısal sütunların birlikte yatay boxplot’u
            fig, ax = plt.subplots(figsize=(15, 6))
            veriler = [st.session_state.df_original[col].dropna() for col in numeric_cols]
            ax.boxplot(veriler, labels=numeric_cols, vert=False)
            ax.set_xlim(0, 5000)
            ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
            ax.set_title("Tüm Sütunlar için Boxplot")
            ax.tick_params(axis="y", rotation=0)
            st.pyplot(fig, use_container_width=False)

        else:
            col_serisi = st.session_state.df_original[seçilen_sütun].dropna()
            Q1 = col_serisi.quantile(0.25)
            Q3 = col_serisi.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask_col = (
                (st.session_state.df_original[seçilen_sütun] < lower)
                | (st.session_state.df_original[seçilen_sütun] > upper)
            )
            outlier_count = mask_col.sum()
            st.write(f"**‘{seçilen_sütun}’ sütununda aykırı değer sayısı:** {outlier_count}")

            fig, ax = plt.subplots(figsize=(4, 2))
            ax.boxplot(col_serisi, vert=True)
            ax.set_title(f"{seçilen_sütun} için Boxplot")
            st.pyplot(fig, use_container_width=False)
    else:
        st.info("Veride sayısal sütun bulunmuyor.")

    remove_out = st.checkbox("Aykırı değerleri IQR yöntemiyle sil", key="remove_outliers")



    if st.button("İşlemi Uygula"):
            
        st.warning("İşlemi Uygula’ butonuna her tıklamada, IQR yöntemi kullanılarak aykırı değerler yeniden tespit edilmektedir; lütfen bu durumu göz önünde bulundurunuz.")
            
        if remove_out:
            # Mevcut veri üzerinden bir kez IQR maskesi hesapla ve uygula
            df_no_outliers = removeOutliers(st.session_state.df_original).reset_index(drop=True)
            removed_count = st.session_state.df_original.shape[0] - df_no_outliers.shape[0]
            st.session_state.df_original = df_no_outliers

            st.success(f"{removed_count} adet aykırı değer içeren satır kaldırıldı. "
                    f"Yeni satır sayısı: {st.session_state.df_original.shape[0]}")

            # Güncellenmiş veri setini tablo olarak göster (indeksler resetlendi)
            st.subheader("Güncel Veri Seti Önizlemesi")
            st.dataframe(st.session_state.df_original)

        else:
            st.info("Aykırı değerleri korumak için önce 'Aykırı değerleri IQR yöntemiyle sil' kutucuğunu işaretleyin.")

    ## ./AYKIRI DEGER ADIMI ##

    st.write("---")

    ## ÇARPIKLIK KONTROLU ##
    numeric_cols = st.session_state.df_original.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.subheader("4. Çarpıklık Kontrolü")
        with st.expander("Çarpıklık (Skewness) Nedir ve Neden Kontrol Edilmelidir?"):
            st.markdown("""
            **Çarpıklık (Skewness) Nedir?**  
            Çarpıklık, bir veri dağılımının simetrideki sapmasını ölçen bir istatistiksel metriktir.  
            - **Pozitif Çarpıklık (Sağa Çarpık):** Dağılımın sağ kuyruk kısmı uzundur ve ortalamadan büyük uç değerler daha fazla görülür.  
            - **Negatif Çarpıklık (Sola Çarpık):** Dağılımın sol kuyruk kısmı uzundur ve ortalamadan küçük uç değerler daha fazla görülür.  

            **Neden Kontrol Edilmelidir?**  
            **Model Performansı:** Aşırı çarpık dağılımlar, birçok makine öğrenmesi veya istatistiksel yöntemde (örneğin lineer regresyon, k-en yakın komşu) modelin öğrenme sürecini olumsuz etkileyebilir. Uç değerlerin (outlier) dengesiz dağılımı, model tahminlerinin yanlılaşmasına yol açabilir.  
            
            **Parametrik Varsayımlar:** Bazı analitik yöntemler, verinin yaklaşık olarak normal (simetrik) dağıldığı varsayımına dayanır. Aşırı çarpık veriler, bu varsayımı ihlal ederek sonuçların güvenilirliğini azaltır.  
            
            **Özellik Dönüşümleri:** Çarpıklığı azaltmak için uygulanacak dönüşümler (örneğin log, kök, Box-Cox, Yeo-Johnson) veriyi daha simetrik hale getirir. Daha simetrik bir dağılım, algoritmaların daha hızlı ve kararlı bir şekilde öğrenmesini sağlar.  
        
            **Özellik Seçimi ve Veri Hazırlığı:** Çok şiddetli çarpık (çok büyük veya çok küçük uç değerlere sahip) sütunlar, model performansına zarar vermemek için dönüştürülemez durumdaysa çıkarılabilir veya farklı bir işleme tabi tutulabilir.  
            
            **Özetle:**  
            Çarpıklık, bir özellikteki değerlerin tek taraflı olarak ağırlık kazanıp kazanmadığını gösterir. Verideki çarpıklık kontrolü ve gerekirse uygun dönüşüm uygulamak, model doğruluğunu ve genelleme yeteneğini artırmak için önemlidir.  
            """)
        
        # Çarpıklık değerlerini hesapla
        skewness = st.session_state.df_original.select_dtypes(include=np.number).skew()

        # Grafik çizimi (Yatay Çubuk)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(skewness.index, skewness.values, color="purple")
        ax.axvline(x=1, color="red", linestyle="--", label="Pozitif eşik (1)")
        ax.axvline(x=-1, color="blue", linestyle="--", label="Negatif eşik (-1)")
        ax.set_title("Özelliklerin Çarpıklık Değerleri (Yatay Çubuk)")
        ax.set_xlabel("Çarpıklık")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        # -------------- SATIR İÇİ SIFIRLAMA + UYARI ----------------
        col1, col2 = st.columns([1, 4])

        with col1:
            if st.button("🔄 Dönüşümleri Sıfırla"):
                if st.session_state.df_backup is not None:
                    st.session_state.df_original = st.session_state.df_backup.copy()
                    st.session_state.df_backup = None
                    st.success("Dönüşüm geri alındı.")
                else:
                    st.info("Geri alınacak bir dönüşüm bulunamadı.")

        with col2:
            with st.expander("Üst üste dönüşüm uygulamasında dikkat edilmelidir."):
                st.warning("""
                **UYARI: Üst Üste Dönüşüm Uygulamak Veri Dağılımını Bozabilir!**

                Veri setinize birden fazla dönüşüm işlemi uygulamak, özellikle çarpıklık giderilmiş sütunlarda **anlamsız sonuçlara veya bilgi kaybına** yol açabilir.  
                - Dönüşüm uygulamadan önce veri yedeğinin (orijinal halinin) alındığından emin olun.  
                - Eğer daha önce dönüşüm uyguladıysanız ve yeni bir dönüşüm planlıyorsanız, **önce dönüşümleri sıfırlamanız önerilir**.

                🔁 *"Dönüşümleri Sıfırla"* butonunu kullanarak veriyi ilk haline getirdikten sonra yeni dönüşüm uygulamanız daha sağlıklı sonuçlar verir.
                """)

        # ------------------------------------------------------------

        yontemler = ["Hiçbiri", "Log Dönüşümü", "Kök Dönüşümü", "Box-Cox Dönüşümü", "Yeo-Johnson Dönüşümü"]
        seçilen_yöntem = st.selectbox("Çarpıklık Azaltma Yöntemi Seçin:", yontemler)

        # Seçilen yönteme göre uyarı/ bilgi metinlerini gösterelim:
        if seçilen_yöntem == "Log Dönüşümü":
            st.warning("""
            **DİKKAT:**  
            - Log dönüşümü yalnızca **sıfır ve pozitif** değerler üzerinde anlamlıdır.  
            - Eğer daha önce Box-Cox ya da başka bir dönüşüm uygulandıysa, veri içinde negatif değerler kalmış olabilir.  
            - Negatif değer içeren sütunlarda `np.log1p` \- uygulaması _“The truth value of a DataFrame is ambiguous”_ veya `NaN/∞` hatalarına yol açabilir.  
            - Log dönüşümü uygulamadan önce veri sütunlarının **tümü** pozitif mi kontrol edilmelidir.
            """)
        elif seçilen_yöntem == "Box-Cox Dönüşümü":
            st.info("""
            **BİLGİ:**  
            - Box-Cox dönüşümü için **girdi verisinin tamamı pozitif** olmalıdır.  
            - Mevcut sütunlarda 0 veya negatif değer varsa, önce `shift = |min|+1` kadar ekleme yapıp, ardından Box-Cox uygulayın.  
            - Aksi halde `PowerTransformer(method="box-cox")` çalıtırken hata alabilirsiniz.
            """)
        elif seçilen_yöntem == "Yeo-Johnson Dönüşümü":
            st.info("""
            **BİLGİ:**  
            - Yeo-Johnson hem pozitif hem negatif değerlere doğrudan uygulanabilir, ekstra “shift” gerektirmez.  
            - Eğer veride negatif değer varsa, Box-Cox yerine Yeo-Johnson tercih edebilirsiniz.
            """)
        elif seçilen_yöntem == "Kök Dönüşümü":
            st.info("""
            **BİLGİ:**  
            - Kök dönüşümü (`sqrt`) için verinin **negatif olmaması** gerekir.  
            """)
        else:
            # “Hiçbiri” seçildiyse bilgi vermeye gerek yok.
            pass

        if st.button("Dönüşüm İşlemini Uygula"):
            if st.session_state.df_backup is None:
                st.session_state.df_backup = st.session_state.df_original.copy()
                
            df_transformed = st.session_state.df_original.copy()
            if seçilen_yöntem == "Log Dönüşümü":
                df_transformed[numeric_cols] = np.log1p(df_transformed[numeric_cols])
                st.success("Log dönüşümü uygulandı.")
            elif seçilen_yöntem == "Kök Dönüşümü":
                df_transformed[numeric_cols] = np.sqrt(df_transformed[numeric_cols].clip(lower=0))
                st.success("Kök dönüşümü uygulandı.")
            elif seçilen_yöntem == "Box-Cox Dönüşümü":
                df_pos = df_transformed[numeric_cols]
                shift = (df_pos.min() <= 0).astype(int) * (abs(df_pos.min()) + 1)
                df_shifted = df_pos + shift
                pt = PowerTransformer(method="box-cox", standardize=False)
                df_transformed[numeric_cols] = pt.fit_transform(df_shifted)
                st.success("Box-Cox dönüşümü uygulandı.")
            elif seçilen_yöntem == "Yeo-Johnson Dönüşümü":
                df_num = df_transformed[numeric_cols]
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                df_transformed[numeric_cols] = pt.fit_transform(df_num)
                st.success("Yeo-Johnson dönüşümü uygulandı.")
            else:
                st.info("Hiçbir dönüşüm uygulanmadı.")
                # Eğer “Hiçbiri” seçildiyse, yedeği de iptal edelim
                st.session_state.df_backup = None

            # Session state'i güncelle ve yeni çarpıklık grafiğini göster
            st.session_state.df_original = df_transformed.reset_index(drop=True)

            new_skewness = st.session_state.df_original.select_dtypes(include=np.number).skew()
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.bar(new_skewness.index, new_skewness.values, color="green")
            ax2.axhline(y=1, color="red", linestyle="--", label="Pozitif eşik (1)")
            ax2.axhline(y=-1, color="blue", linestyle="--", label="Negatif eşik (-1)")
            ax2.set_title("Dönüşüm Sonrası Çarpıklık Değerleri")
            ax2.set_ylabel("Çarpıklık")
            ax2.legend()
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig2)

            st.subheader("Güncel Veri Seti Önizlemesi (Dönüşüm Uygulandı)")
            st.dataframe(st.session_state.df_original)

    else:
        st.info("Çarpıklık kontrolü için sayısal sütun bulunamadı.")
    ## ./ÇARPIKLIK KONTROLU ##

    st.write("---")

    ## KORELASYON ADIMI ##
    st.subheader("5. Korelasyon Analizi")

    with st.expander("Korelasyon Nedir ve Neden İşlenmelidir?"):
        st.markdown("""
        **Korelasyon Nedir?**
        - İki sayısal değişken arasındaki doğrusal ilişkiyi ölçer.  
        - Değer aralığı **-1** (tam negatif ilişki) ile **+1** (tam pozitif ilişki) arasındadır.  
        - \|r\| arttıkça ilişki gücü artar; \(r=0\) doğrusal ilişki olmadığını gösterir.

        **Neden Özellik Seçimi Yapılmalı?**
        1. **Çoklu Doğrusal Bağıntıyı Önleme:**  
        Yüksek korelasyonlu değişkenler multicollinearity oluşturarak model katsayılarını kararsızlaştırır.  
        2. **Model Basitliği ve Yorumlanabilirlik:**  
        Az sayıda, fakat bilgi yönünden zengin özelliklerle çalışmak hem anlaşılır hem de sürdürülebilirdir.  
        3. **Aşırı Öğrenmeyi Azaltma (Overfitting):**  
        Yinelenen sinyaller modelin eğitim verisine fazla uyum sağlamasına yol açabilir.  
        4. **Hesaplama Verimliliği:**  
        Daha az özellik, daha hızlı eğitim ve tahmin süresi demektir.

        **Kaynakça**
        - Rodgers, J. L., & Nicewander, W. A. (1988). Thirteen Ways to Look at the Correlation Coefficient. *The American Statistician*, 42(1), 59–66.  
        """)


    # 1. Sayısal sütunları al
    df_num = st.session_state.df_original.select_dtypes(include=['float64', 'int64'])

    # 2. Korelasyon matrisi ve heatmap
    corr = df_num.corr().abs()
    fig, ax = plt.subplots(figsize=(20, 12))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Korelasyon Matrisi (|r|)", fontsize=16)
    st.pyplot(fig)

    # ——— Grafiğin hemen altı ———

    # 3. Eşik değerini seçmek için slider
    threshold = st.slider(
        "Korelasyon Eşiği (|r|)",    # label
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.01
    )

    # 4. Yüksek korelasyona sahip çiftleri bul
    high_corr_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .loc[lambda x: x > threshold]
            .sort_values(ascending=False)
    )

    if not high_corr_pairs.empty:
        # DataFrame’e çevir ve sabit yükseklikte göster
        df_pairs = high_corr_pairs.reset_index()
        df_pairs.columns = ["Feature A", "Feature B", "|r|"]
        st.dataframe(df_pairs, height=300)   # ← burada yükseklik ve scrollbar
    else:
        st.info(f"|r| > {threshold:.2f} değerinde hiç çift bulunamadı.")

    # 5. Silinecek sütunları seçme kutusu
    to_drop = st.multiselect(
        "Korelasyonu yüksek değişkenleri silmek için seçin",
        options=sorted({feat for pair in high_corr_pairs.index for feat in pair}),
        help="Seçilen sütunlar veri setinden tamamen kaldırılacak."
    )

    # 6. Silme butonu
    if st.button("Seçilen Özellikleri Sil"):
        if to_drop:
            st.session_state.df_original.drop(columns=to_drop, inplace=True)
            st.success(f"{len(to_drop)} sütun kaldırıldı: {', '.join(to_drop)}")
            st.subheader("Güncel Veri Seti Önizlemesi")
            st.dataframe(st.session_state.df_original)
        else:
            st.info("Silinecek sütun seçilmedi.")
    ## ./KORELASYON ADIMI ##

    st.write('---')

    ## VERI SETI DISA AKTAR ##
    st.subheader("6. İşlenen Veri Setini CSV Olarak İndir")
    # DataFrame'i CSV’ye çevir ve byte dizisine encode et
    csv_data = st.session_state.df_original.to_csv(index=False).encode("utf-8")
    # İndir butonu
    download_clicked = st.download_button(
        label="📥 CSV Olarak İndir",
        data=csv_data,
        file_name="islenmis_veriset.csv",
        mime="text/csv"
    )

    if download_clicked:
            st.warning(
                "İndirdiğiniz dosya hassas bilgileri içerebilir. "
                "Paylaşmadan önce kişisel verileri anonimleştirdiğinizden emin olun."
            )
    ##./ VERI SETI DISA AKTAR ##

    ### BASLANGIC (END) ###
