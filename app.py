### KUTUPHANELER (START) ###
import streamlit as st
st.set_page_config(page_title="Bitirme Projesi | Baran Kanat", layout="wide")
from baslangic import app as baslangic_app
from veriAnalizi import app as veri_app
from modelAnalizi import app as model_app
from dinamikVeri import app as dinamik_app
### KUTUPHANELER (END) ###



st.sidebar.title("Yapay Zeka Analizi")

# Menü seçeneklerini tanımla; "Başlangıç" ilk seçenek ve varsayılan
menu = [
    "Başlangıç",
    "Dinamik Veri Analizi",
    "Veri Analizi & Ön İşleme",
    "Model Eğitimi & Değerlendirilmesi"
]
choice = st.sidebar.radio("Projem", menu, index=0)

# Seçime göre ilgili fonksiyonu çalıştır
if choice == "Başlangıç":
    baslangic_app()
elif choice == "Dinamik Veri Analizi":
    dinamik_app()
elif choice == "Veri Analizi & Ön İşleme":
    veri_app()
elif choice == "Model Eğitimi & Değerlendirilmesi":
    model_app()
