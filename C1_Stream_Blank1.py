# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:41:35 2026

@author: monka
"""


#otwieranie Streamlite'a
#!python -m streamlit run C1_stream_Blank1.py --server.headless true --browser.gatherUsageStats false


#!/usr/bin/env python
# -*- coding: utf-8 -*-
#pomoc: https://www.kaggle.com/code/stetelepta/exploring-heart-rate-variability-using-python

#%%------------------------------------BIBLIOTEKI------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import glob, os
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import warnings
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import neurokit2 as nk
import gdown

#%%--------------------------------Ustawienia wstępne kolorów------------------

st.set_page_config(layout="wide")

#---------------------------Definicje Kolorów do których potem się odwołujemy

bialy          ="#ffffff"
bialo_szary    ="#aeaeae"
rozowy         ="#ff29a7"
niebieski      ="#0092ff"
zielony        ="#41c232"
czerwony       ="#ff1100"
lekki_szary    ="#363636"
lekki_czerwony ="#e74c3c"
mocny_szary    ="#999999"
czarny         ="#000000"
charcoal       ="#36454F"

#---------------------------Ustawienia kolorów

st.markdown(f"""
    <style>
    /* 1. Zmiana koloru głównego tekstu w aplikacji */
    .stApp {{
        color: {czarny};
    }}

    /* 2. Zmiana koloru dla wszystkich nagłówków (h1, h2, h3) */
    h1, h2, h3, [data-testid="stHeader"] {{
        color: {bialy} !important;
    }}

    /* 3. Zmiana koloru zwykłego tekstu (paragrafy, opisy) */
    p, .stText, [data-testid="stWidgetLabel"] {{
        color: {charcoal};
        font-size: 16px;
    }}

    /* 4. Metryki - Wartość (liczba) */
    [data-testid="stMetricValue"] {{
        font-size: 18px !important;
        color: {bialy} !important; 
    }}
    
    /* 5. Metryki - Etykieta (opis nad liczbą) */
    [data-testid="stMetricLabel"] p {{
        color: {mocny_szary} !important;
    }}
    </style>
    """, unsafe_allow_html=True)
    
#%%---------------------------------Tytuł i ramka------------------------------

st.markdown(f"""
    <style>
    .moja-ramka {{
        
        border-radius: 10px;
        padding: 20px;
        background-color: {lekki_szary};
        text-align: center;
        height: 120px;
    }}
    .moja-ramka h4 {{
        color: {lekki_czerwony};
        margin: 0;
    }}
    </style>
    
    <div class="moja-ramka">
        <h4>Analiza HRV sygnału EKG</h4>
        <p style="color: {bialy};">Laboratorium z biofizyki dla fizyków</p>
    </div>
    """, unsafe_allow_html=True)   

#---------------------------Pozioma linia biała

st.markdown(f"""
    <hr style="margin-top: 10px;height:5px; border:none; color:{lekki_szary}; background-color:#444444;" />
""", unsafe_allow_html=True)


@st.cache_data
def load_my_data():
    filename_1 = "ekg_400Hz_10min.txt"
    file_id1 = '1UsK27t-StzlXlUE6vPmmQ19bcmokHZ_p'

    filename_2 = "ekg_400Hz_10min_wysilkowe.txt"
    file_id2 = "1De3jA04yf-FoIqV9hToVFKY9RgzCJI2H"


    # Funkcja pomocnicza do pobierania
    def download_file(file_id, output_name):
        url = f'https://drive.google.com/uc?id={file_id}'
        # Jeśli plik istnieje, usuwamy go, aby gdown pobrał nową wersję
        if os.path.exists(output_name):
            os.remove(output_name)

        try:
            # gdown czasami potrzebuje parametru fuzzy=True dla GDrive
            gdown.download(url, output_name, quiet=False)
        except Exception as e:
            st.error(f"Błąd pobierania {output_name}: {e}")

 

    # Pobieranie plików
    with st.spinner('Synchronizacja z Google Drive...'):
        download_file(file_id1, filename_1)
        download_file(file_id2, filename_2)


 

    # Wczytywanie
    try:

        data = pd.read_csv(filename_1, sep='\t', decimal=',', header=None, skiprows=10)
        data = data.apply(pd.to_numeric, errors='coerce')
        data = data.dropna()
        data2 = pd.read_csv(filename_2, sep='\t', decimal=',', header=None, skiprows=10)
        data2 = data2.apply(pd.to_numeric, errors='coerce')
        data2 = data2.dropna()
        return data, data2
    except Exception as e:
        st.error(f"Błąd wczytywania pkl: {e}")
        return pd.DataFrame(), pd.DataFrame()

 

# Wywołanie danych
df_spocz, df_wys = load_my_data()
"""def sload_my_data(file):
    data = pd.read_csv(file, sep='\t', decimal=',', header=None, skiprows=10)
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    return data
    """

# ============================================================
# WCZYTANIE DANYCH
# ============================================================



fs = 400  # Hz

# Nadajemy kolumny od razu po wczytaniu (zakładamy 3 kolumny: czas, oddech, ecg)
df_spocz.columns = ['czas', 'ecg']
df_wys.columns   = ['czas', 'ecg']

# Czas z indeksu (nadpisujemy kolumnę czas jeśli plik nie ma własnego czasu)
df_spocz['czas'] = df_spocz.index / fs
df_wys['czas']   = df_wys.index / fs

signal_col = 'ecg'  # teraz używamy nazwy kolumny, nie liczby

#%%-----------------------------SEKCJA 1 - ZAKRES SYGNAŁU-SPOCZYNKOWE

col1, col2, col3 = st.columns([2.0,1.5,5])
    
with col1:
    
#---------------------------Kolumna 1 
    st.subheader("Spoczynkowe")        
    st.dataframe(df_spocz,height=265, use_container_width=True)

with col2:

    #-----------------------Kolumna 2
            
    min_czas = float(df_spocz['czas'].min())
    max_czas = float(df_spocz['czas'].max())

    zakres_czasu = st.slider(
        "Wybierz zakres czasu do analizy [s]:",
        min_value=min_czas,
        max_value=max_czas,
        value=(min_czas, max_czas),
        step=0.1,
        key="slider_spocz"      # ← klucz ważny gdy będziesz miał drugi suwak dla wysiłkowego
    )

    df_stary_spocz = df_spocz.copy()
    df_spocz = df_spocz[
        (df_spocz['czas'] >= zakres_czasu[0]) &
        (df_spocz['czas'] <= zakres_czasu[1])
    ].copy()

    ile_zostalo = len(df_spocz)
    ile_wycieto = len(df_stary_spocz) - ile_zostalo

    dane_pie = {
        "Status": ["Fragment do analizy", "Pozostała część"],
        "Liczba próbek": [ile_zostalo, ile_wycieto]
    }

# 3. Tworzenie wykresu Plotly Express
    fig_pie = px.pie(
        dane_pie,
        values='Liczba próbek',
        names='Status',
        hole=0.4,
        color_discrete_sequence=['#e74c3c', '#7e7e7e']
    )

# 4. Stylizacja
    fig_pie.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=0, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
with col3:
        
    fig = go.Figure()

        # 2. Dodajemy Sygnał Surowy (niebieski, cieńszy)
    fig.add_trace(go.Scatter(
            x=df_stary_spocz['czas'], 
            y=df_stary_spocz[signal_col], 
            mode='lines',
            name='Pozostała częsc',
            line=dict(color=bialo_szary, width=2)
        ))

        # 3. Dodajemy Sygnał Przefiltrowany (czerwony, grubszy)
    fig.add_trace(go.Scatter(
            x=df_spocz['czas'], 
            y=df_spocz[signal_col], 
            mode='lines',
            name='Fragment do analizy',
            line=dict(color=lekki_czerwony, width=3) # Wyrazisty czerwony
        ))

        # 4. Stylizacja wykresu
    fig.update_layout(
            height=230,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",      # Legenda w poziomie
                yanchor="bottom",
                y=1.02,               # Nad wykresem
                xanchor="left",       # Zakotwiczenie do lewej
                x=0                   # Pozycja na osi X (0 = start od lewej)
            ),
            xaxis_title="Czas [s]",
            yaxis_title="Amplituda [mV]"
        )

    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)
    # Dodajemy grubszą linię dla oddzielenia (tę, którą robiliśmy wcześniej)


    
st.markdown("""
    <hr style="margin-top: -10px;height:5px; border:none; color:#444444; background-color:#444444;" />
""", unsafe_allow_html=True)

#%%------------------------------------------------------------------


#%%-----------------------------SEKCJA 1 - ZAKRES SYGNAŁU-WYSIŁKOWE

col1, col2, col3 = st.columns([2.0,1.5,5])
    
with col1:
    
#---------------------------Kolumna 1 
    st.subheader("Wysiłkowe")        
    st.dataframe(df_wys,height=265, use_container_width=True)

with col2:

    #-----------------------Kolumna 2
            
    min_czas = float(df_wys['czas'].min())
    max_czas = float(df_wys['czas'].max())

    zakres_czasu = st.slider(
        "Wybierz zakres czasu do analizy [s]:",
        min_value=min_czas,
        max_value=max_czas,
        value=(min_czas, max_czas),
        step=0.1,
        key="slider_wys"      # ← klucz ważny gdy będziesz miał drugi suwak dla wysiłkowego
    )

    df_stary_wys = df_wys.copy()
    df_wys = df_wys[
        (df_wys['czas'] >= zakres_czasu[0]) &
        (df_wys['czas'] <= zakres_czasu[1])
    ].copy()

    ile_zostalo = len(df_wys)
    ile_wycieto = len(df_stary_wys) - ile_zostalo

    dane_pie = {
        "Status": ["Fragment do analizy", "Pozostała część"],
        "Liczba próbek": [ile_zostalo, ile_wycieto]
    }

# 3. Tworzenie wykresu Plotly Express
    fig_pie = px.pie(
        dane_pie,
        values='Liczba próbek',
        names='Status',
        hole=0.4,
        color_discrete_sequence=['#41c232', '#7e7e7e']
    )

# 4. Stylizacja
    fig_pie.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=0, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig_pie, use_container_width=True)
    
with col3:
        
    fig = go.Figure()

        # 2. Dodajemy Sygnał Surowy (niebieski, cieńszy)
    fig.add_trace(go.Scatter(
            x=df_stary_wys['czas'], 
            y=df_stary_wys[signal_col], 
            mode='lines',
            name='Pozostała częsc',
            line=dict(color=bialo_szary, width=2)
        ))

        # 3. Dodajemy Sygnał Przefiltrowany (czerwony, grubszy)
    fig.add_trace(go.Scatter(
            x=df_wys['czas'], 
            y=df_wys[signal_col], 
            mode='lines',
            name='Fragment do analizy',
            line=dict(color=zielony, width=3) # Wyrazisty czerwony
        ))

        # 4. Stylizacja wykresu
    fig.update_layout(
            height=230,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",      # Legenda w poziomie
                yanchor="bottom",
                y=1.02,               # Nad wykresem
                xanchor="left",       # Zakotwiczenie do lewej
                x=0                   # Pozycja na osi X (0 = start od lewej)
            ),
            xaxis_title="Czas [s]",
            yaxis_title="Amplituda [mV]"
        )

    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)
    # Dodajemy grubszą linię dla oddzielenia (tę, którą robiliśmy wcześniej)


    
st.markdown("""
    <hr style="margin-top: -10px;height:5px; border:none; color:#444444; background-color:#444444;" />
""", unsafe_allow_html=True)

#%%-----------------------------SEKCJA 2 - FILTRY------------------------------

st.markdown(f"""
    <hr style="margin-top: 10px;height:5px; border:none; color:#444444; background-color:#444444;" />
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 7])

with col1:
    # ---- SPOCZYNKOWE - suwaki ----
    st.markdown(f"""
        <div style="background-color: {lekki_czerwony}; 
            border-radius: 10px; 
            padding: 20px;
            margin-bottom: 10px;
        ">
        <p style="color:white; font-weight:bold; text-align:center; margin:0;">Filtrowanie – spoczynkowe</p>
        </div>
    """, unsafe_allow_html=True)

    lewy, srodek, prawy = st.columns([0.1, 0.8, 0.1])
    with srodek:
        window_length_spocz = st.slider("Długość okna filtra:", min_value=1, max_value=102, value=43, step=2, key="wl_spocz")
        polyorder_spocz     = st.slider("Stopień wielomianu:",  min_value=1, max_value=6,   value=2,  step=1, key="po_spocz")

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    # ---- WYSIŁKOWE - suwaki ----
    st.markdown(f"""
        <div style="background-color: {zielony}; 
            border-radius: 10px; 
            padding: 20px;
            margin-bottom: 10px;
        ">
        <p style="color:white; font-weight:bold; text-align:center; margin:0;">Filtrowanie – wysiłkowe</p>
        </div>
    """, unsafe_allow_html=True)

    lewy, srodek, prawy = st.columns([0.1, 0.8, 0.1])
    with srodek:
        window_length_wys = st.slider("Długość okna filtra:", min_value=1, max_value=102, value=43, step=2, key="wl_wys")
        polyorder_wys     = st.slider("Stopień wielomianu:",  min_value=1, max_value=6,   value=2,  step=1, key="po_wys")

# ---- Filtrowanie ----
df_spocz['ecg_filtrowany'] = savgol_filter(df_spocz['ecg'].astype(float).values, window_length_spocz, polyorder_spocz)
df_wys['ecg_filtrowany']   = savgol_filter(df_wys['ecg'].astype(float).values,   window_length_wys,   polyorder_wys)

with col2:
    # ---- WYKRES SPOCZYNKOWY ----
    st.markdown(f"###### Filtracja sygnału – spoczynkowe")
    fig_spocz = go.Figure()
    fig_spocz.add_trace(go.Scatter(
        x=df_spocz['czas'], y=df_spocz['ecg'],
        mode='lines', name='Surowy',
        line=dict(color='rgba(52, 152, 219, 0.5)', width=1)
    ))
    fig_spocz.add_trace(go.Scatter(
        x=df_spocz['czas'], y=df_spocz['ecg_filtrowany'],
        mode='lines', name='Savgol Filter',
        line=dict(color=lekki_czerwony, width=3)
    ))
    fig_spocz.update_layout(
        height=232,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Czas [s]",
        yaxis_title="Amplituda [mV]"
    )
    with st.container(border=True):
        st.plotly_chart(fig_spocz, use_container_width=True)

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    # ---- WYKRES WYSIŁKOWY ----
    st.markdown(f"###### Filtracja sygnału – wysiłkowe")
    fig_wys = go.Figure()
    fig_wys.add_trace(go.Scatter(
        x=df_wys['czas'], y=df_wys['ecg'],
        mode='lines', name='Surowy',
        line=dict(color='rgba(52, 152, 219, 0.5)', width=1)
    ))
    fig_wys.add_trace(go.Scatter(
        x=df_wys['czas'], y=df_wys['ecg_filtrowany'],
        mode='lines', name='Savgol Filter',
        line=dict(color=zielony, width=3)
    ))
    fig_wys.update_layout(
        height=232,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Czas [s]",
        yaxis_title="Amplituda [mV]"
    )
    with st.container(border=True):
        st.plotly_chart(fig_wys, use_container_width=True)
#%%-----------------------------SEKCJA 3 - Załamki R (SPOCZYNKOWE)--------------

st.markdown("""
    <hr style="margin-top: 10px;height:5px; border:none; color:#444444; background-color:#444444;" />
""", unsafe_allow_html=True)

st.markdown(f'<p style="margin-top: 0px; font-size: 18px; font-weight: bold; color:{lekki_czerwony};">Identyfikacja załamków R – sygnał spoczynkowy</p>', unsafe_allow_html=True)

col1, col2 = st.columns([4, 4.5])

with col1:
    col_left, col_right = st.columns([1, 4])

    with col_left:
        st.markdown(f"""
            <div style="background-color: {lekki_czerwony}; 
                border-radius: 10px; 
                padding: 40px;
                margin-bottom: -1820px;
                height: 230px;
                border: 0px solid rgba(100,100,100,1);
            ">
            </div>
        """, unsafe_allow_html=True)

        lewy, srodek, prawy = st.columns([0.1, 0.9, 0.1])
        with srodek:
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
            threshold_rr_spocz = st.slider("Próg dla pików R:", min_value=0.0, max_value=2.0, value=0.11, step=0.01, key="thr_spocz")
            distance_rr_spocz  = st.slider("Dystans między RR:", min_value=0.0, max_value=2000.0, value=450.0, step=10.0, key="dist_spocz")

        sygnal_spocz = df_spocz['ecg_filtrowany'].values
        peaks_spocz, _ = find_peaks(sygnal_spocz, distance=distance_rr_spocz, height=threshold_rr_spocz)

    with col_right:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_spocz['czas'],
            y=df_spocz['ecg_filtrowany'],
            mode='lines', name='Sygnał EKG',
            line=dict(color='#FFBCBC', width=1.5)
        ))

        fig.add_trace(go.Scatter(
            x=df_spocz['czas'].iloc[peaks_spocz],
            y=df_spocz['ecg_filtrowany'].iloc[peaks_spocz],
            mode='markers', name='Piki R',
            marker=dict(color=lekki_czerwony, size=8, symbol='circle',
                        line=dict(color='white', width=1))
        ))

        fig.add_hline(y=threshold_rr_spocz, line_dash="dash",
                      line_color="rgba(255,255,255,0.3)",
                      annotation_text="Aktualny próg")

        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Czas [s]",
            yaxis_title="Amplituda [mV]"
        )

        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)

    # ---- Tachogram RR ----
    czasy_pikow_spocz = df_spocz['czas'].iloc[peaks_spocz].values
    odstepy_rr_spocz  = np.diff(czasy_pikow_spocz)

    df_rr_spocz = pd.DataFrame({
        '#':     range(1, len(odstepy_rr_spocz) + 1),
        'rr_ms': odstepy_rr_spocz * 1000,
        'rr_s':  odstepy_rr_spocz
    })

    st.markdown(f"""
        <div style="background-color: {lekki_szary}; 
            border-radius: 10px; padding: 40px;
            margin-bottom: -1820px;
            height: 300px;
            border: 0px solid rgba(100,100,100,1);">
        </div>
    """, unsafe_allow_html=True)

    lewy, srodek, prawy = st.columns([0.02, 0.9, 0.02])
    with srodek:
        fig_tach = go.Figure()

        fig_tach.add_trace(go.Scatter(
            x=df_spocz['czas'].iloc[peaks_spocz],
            y=df_rr_spocz['rr_ms'].values,
            mode='lines+markers',
            name='Odstępy RR',
            line=dict(color=bialy, width=2),
            marker=dict(size=6, color=lekki_czerwony, symbol='circle')
        ))

        fig_tach.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Czas badania [s]",
            yaxis_title="Odstęp RR [ms]",
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=30, r=20, t=30, b=90),
            height=300
        )

        st.plotly_chart(fig_tach, use_container_width=True)

with col2:
    st.markdown(f'<p style="margin-top: 0px; font-size: 18px; font-weight: bold; color:{lekki_czerwony};">Histogram – spoczynkowe</p>', unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background-color: {lekki_szary}; 
            border-radius: 10px; padding: 40px;
            margin-bottom: -1820px;
            height: 550px;
            border: 0px solid rgba(100,100,100,1);">
        </div>
    """, unsafe_allow_html=True)

    lewy, srodek, prawy = st.columns([0.02, 0.9, 0.02])
    with srodek:
        col_rr1, col_rr2 = st.columns([1., 1.8])

        with col_rr1:
            st.dataframe(df_rr_spocz, height=310, use_container_width=True)

        with col_rr2:
            histogram_bins_spocz = st.slider('Histogram', min_value=20, max_value=300, value=180, step=1, key="hist_spocz")

            fig_hist = px.histogram(
                df_rr_spocz,
                x="rr_ms",
                nbins=histogram_bins_spocz,
                labels={'rr_ms': 'Odstęp RR [ms]'},
                color_discrete_sequence=[lekki_czerwony],
                marginal="rug"
            )

            fig_hist.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Czas trwania [ms]",
                yaxis_title="Częstość",
                bargap=0.1
            )

            with st.container(border=True):
                st.plotly_chart(fig_hist, use_container_width=True)

        # ---- Metryki ----
        srednie_rr = df_rr_spocz['rr_ms'].mean()
        sdnn       = df_rr_spocz['rr_ms'].std()
        max_rr     = df_rr_spocz['rr_ms'].max()
        min_rr     = df_rr_spocz['rr_ms'].min()
        liczba_R   = df_rr_spocz.shape[0]

        st.markdown(f"""
            <hr style="margin-top: 10px;height:5px; border:none; background-color:{lekki_czerwony};" />
        """, unsafe_allow_html=True)

        cola, colb, colc, cold, cole = st.columns([1, 1, 1, 1, 2])
        with cola:
            st.metric("Średnie RR", f"{srednie_rr:.0f} ms")
        with colb:
            st.metric("Std RR", f"{sdnn:.0f} ms")
        with colc:
            st.metric("Max RR", f"{max_rr:.0f} ms")
        with cold:
            st.metric("Min RR", f"{min_rr:.0f} ms")
        with cole:
            st.metric("Liczba zidentyfikowanych załamków R", f"{liczba_R:.0f}")

        st.markdown(f"""
            <hr style="margin-top: 10px;height:5px; border:none; background-color:{lekki_czerwony};" />
        """, unsafe_allow_html=True)
        
        
#%%-----------------------------SEKCJA 3 - Załamki R (WYSIŁKOWE)----------------

st.markdown("""
    <hr style="margin-top: 10px;height:5px; border:none; color:#444444; background-color:#444444;" />
""", unsafe_allow_html=True)

st.markdown(f'<p style="margin-top: 0px; font-size: 18px; font-weight: bold; color:{zielony};">Identyfikacja załamków R – sygnał wysiłkowy</p>', unsafe_allow_html=True)

col1, col2 = st.columns([4, 4.5])

with col1:
    col_left, col_right = st.columns([1, 4])

    with col_left:
        st.markdown(f"""
            <div style="background-color: {zielony}; 
                border-radius: 10px; 
                padding: 40px;
                margin-bottom: -1820px;
                height: 230px;
                border: 0px solid rgba(100,100,100,1);
            ">
            </div>
        """, unsafe_allow_html=True)

        lewy, srodek, prawy = st.columns([0.1, 0.9, 0.1])
        with srodek:
            st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
            threshold_rr_wys = st.slider("Próg dla pików R:", min_value=0.0, max_value=2.0, value=0.11, step=0.01, key="thr_wys")
            distance_rr_wys  = st.slider("Dystans między RR:", min_value=0.0, max_value=2000.0, value=450.0, step=10.0, key="dist_wys")

        sygnal_wys = df_wys['ecg_filtrowany'].values
        peaks_wys, _ = find_peaks(sygnal_wys, distance=distance_rr_wys, height=threshold_rr_wys)

    with col_right:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_wys['czas'],
            y=df_wys['ecg_filtrowany'],
            mode='lines', name='Sygnał EKG',
            line=dict(color='#e0ffb0', width=1.5)
        ))

        fig.add_trace(go.Scatter(
            x=df_wys['czas'].iloc[peaks_wys],
            y=df_wys['ecg_filtrowany'].iloc[peaks_wys],
            mode='markers', name='Piki R',
            marker=dict(color=zielony, size=8, symbol='circle',
                        line=dict(color='white', width=1))
        ))

        fig.add_hline(y=threshold_rr_wys, line_dash="dash",
                      line_color="rgba(255,255,255,0.3)",
                      annotation_text="Aktualny próg")

        fig.update_layout(
            height=200,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Czas [s]",
            yaxis_title="Amplituda [mV]"
        )

        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True)

    # ---- Tachogram RR ----
    czasy_pikow_wys = df_wys['czas'].iloc[peaks_wys].values
    odstepy_rr_wys  = np.diff(czasy_pikow_wys)

    df_rr_wys = pd.DataFrame({
        '#':     range(1, len(odstepy_rr_wys) + 1),
        'rr_ms': odstepy_rr_wys * 1000,
        'rr_s':  odstepy_rr_wys
    })

    st.markdown(f"""
        <div style="background-color: {lekki_szary}; 
            border-radius: 10px; padding: 40px;
            margin-bottom: -1820px;
            height: 300px;
            border: 0px solid rgba(100,100,100,1);">
        </div>
    """, unsafe_allow_html=True)

    lewy, srodek, prawy = st.columns([0.02, 0.9, 0.02])
    with srodek:
        fig_tach = go.Figure()

        fig_tach.add_trace(go.Scatter(
            x=df_wys['czas'].iloc[peaks_wys],
            y=df_rr_wys['rr_ms'].values,
            mode='lines+markers',
            name='Odstępy RR',
            line=dict(color=bialy, width=2),
            marker=dict(size=6, color=zielony, symbol='circle')
        ))

        fig_tach.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Czas badania [s]",
            yaxis_title="Odstęp RR [ms]",
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=30, r=20, t=30, b=90),
            height=300
        )

        st.plotly_chart(fig_tach, use_container_width=True)

with col2:
    st.markdown(f'<p style="margin-top: 0px; font-size: 18px; font-weight: bold; color:{zielony};">Histogram – wysiłkowe</p>', unsafe_allow_html=True)

    st.markdown(f"""
        <div style="background-color: {lekki_szary}; 
            border-radius: 10px; padding: 40px;
            margin-bottom: -1820px;
            height: 550px;
            border: 0px solid rgba(100,100,100,1);">
        </div>
    """, unsafe_allow_html=True)

    lewy, srodek, prawy = st.columns([0.02, 0.9, 0.02])
    with srodek:
        col_rr1, col_rr2 = st.columns([1., 1.8])

        with col_rr1:
            st.dataframe(df_rr_wys, height=310, use_container_width=True)

        with col_rr2:
            histogram_bins_wys = st.slider('Histogram', min_value=20, max_value=300, value=180, step=1, key="hist_wys")

            fig_hist = px.histogram(
                df_rr_wys,
                x="rr_ms",
                nbins=histogram_bins_wys,
                labels={'rr_ms': 'Odstęp RR [ms]'},
                color_discrete_sequence=[zielony],
                marginal="rug"
            )

            fig_hist.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Czas trwania [ms]",
                yaxis_title="Częstość",
                bargap=0.1
            )

            with st.container(border=True):
                st.plotly_chart(fig_hist, use_container_width=True)

        # ---- Metryki ----
        srednie_rr = df_rr_wys['rr_ms'].mean()
        sdnn       = df_rr_wys['rr_ms'].std()
        max_rr     = df_rr_wys['rr_ms'].max()
        min_rr     = df_rr_wys['rr_ms'].min()
        liczba_R   = df_rr_wys.shape[0]

        st.markdown(f"""
            <hr style="margin-top: 10px;height:5px; border:none; background-color:{zielony};" />
        """, unsafe_allow_html=True)

        cola, colb, colc, cold, cole = st.columns([1, 1, 1, 1, 2])
        with cola:
            st.metric("Średnie RR", f"{srednie_rr:.0f} ms")
        with colb:
            st.metric("Std RR", f"{sdnn:.0f} ms")
        with colc:
            st.metric("Max RR", f"{max_rr:.0f} ms")
        with cold:
            st.metric("Min RR", f"{min_rr:.0f} ms")
        with cole:
            st.metric("Liczba zidentyfikowanych załamków R", f"{liczba_R:.0f}")

        st.markdown(f"""
            <hr style="margin-top: 10px;height:5px; border:none; background-color:{zielony};" />
        """, unsafe_allow_html=True)
        
        
#%%-----------------------------SEKCJA 4 - Segmentacja QRS (SPOCZYNKOWE)--------

st.markdown(f"""
    <hr style="margin-top: 10px;height:5px; border:none; color:#444444; background-color:#444444;" />
""", unsafe_allow_html=True)

st.markdown(f'<p style="margin-top: 0px; font-size: 18px; font-weight: bold; color:{lekki_czerwony};">Segmentacja zespołu QRS – sygnał spoczynkowy</p>', unsafe_allow_html=True)

st.markdown(f"""
    <hr style="margin-top: 10px;height:5px; border:none; background-color:{lekki_czerwony};" />
""", unsafe_allow_html=True)

col1, col2 = st.columns([4, 4.5])

with col1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        window_spocz = st.slider("Szerokość okna [próbki]:", min_value=100, max_value=1000, value=250, step=10, key="window_spocz")

        ecg_signal_spocz = df_spocz['ecg_filtrowany'].values
        qrs_dict_spocz = {}

        for i, r in enumerate(peaks_spocz):
            if r > window_spocz and r + window_spocz < len(ecg_signal_spocz):
                segment = ecg_signal_spocz[r - window_spocz : r + window_spocz].copy()
                segment = segment - np.mean(segment)
                qrs_dict_spocz[f'QRS_{i+1:02d}'] = segment

        df_qrs_spocz = pd.DataFrame(qrs_dict_spocz, index=range(-window_spocz, window_spocz))

    with col_right:
        idx_segmentu_spocz = st.slider("Wybierz numer zespołu QRS:", 0, len(peaks_spocz)-1, 2, key="idx_spocz")

    # Podgląd segmentu na tle całego sygnału
    r_center_spocz = peaks_spocz[idx_segmentu_spocz]
    start_idx_spocz = max(0, r_center_spocz - window_spocz)
    stop_idx_spocz  = min(len(df_spocz), r_center_spocz + window_spocz)

    fig_seg_spocz = go.Figure()

    fig_seg_spocz.add_trace(go.Scatter(
        x=df_spocz['czas'],
        y=df_spocz['ecg_filtrowany'],
        mode='lines',
        line=dict(color='rgba(150, 150, 150, 0.5)', width=1),
        name='Pełny sygnał'
    ))

    fig_seg_spocz.add_trace(go.Scatter(
        x=df_spocz['czas'].iloc[start_idx_spocz:stop_idx_spocz],
        y=df_spocz['ecg_filtrowany'].iloc[start_idx_spocz:stop_idx_spocz],
        mode='lines',
        line=dict(color=lekki_czerwony, width=3),
        name='Wybrany segment'
    ))

    fig_seg_spocz.add_vrect(
        x0=df_spocz['czas'].iloc[start_idx_spocz],
        x1=df_spocz['czas'].iloc[stop_idx_spocz],
        fillcolor=lekki_czerwony, opacity=0.2,
        layer="below", line_width=0,
    )

    fig_seg_spocz.update_layout(
        title=f"Podgląd segmentu nr {idx_segmentu_spocz + 1} (R w {df_spocz['czas'].iloc[r_center_spocz]:.2f} s)",
        xaxis_title="Czas [s]",
        yaxis_title="Amplituda",
        template="plotly_dark",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    st.plotly_chart(fig_seg_spocz, use_container_width=True)

with col2:
    lewy, prawy = st.columns([1, 1])

    with lewy:
        df_qrs_spocz['SREDNI_QRS'] = df_qrs_spocz.mean(axis=1)
        y_min_s = df_qrs_spocz['SREDNI_QRS'].min()
        y_max_s = df_qrs_spocz['SREDNI_QRS'].max()
        margines_s = (y_max_s - y_min_s) * 0.15

        fig_qrs_spocz = px.line(df_qrs_spocz,
                                labels={'index': 'Próbki względem R', 'value': 'Amplituda'},
                                title="Nałożone segmenty QRS z uśrednionym profilem")

        fig_qrs_spocz.update_traces(line=dict(width=1, color='rgba(150, 150, 150, 0.3)'), opacity=0.4)
        fig_qrs_spocz.update_layout(
            yaxis=dict(range=[y_min_s - margines_s, y_max_s + margines_s], fixedrange=False),
            template="plotly_dark",
            uirevision='constant'
        )

        fig_qrs_spocz.for_each_trace(
            lambda trace: trace.update(line=dict(color=lekki_czerwony, width=4), opacity=1)
            if trace.name == 'SREDNI_QRS' else ()
        )

        fig_qrs_spocz.data = [t for t in fig_qrs_spocz.data if t.name != 'SREDNI_QRS'] + \
                             [t for t in fig_qrs_spocz.data if t.name == 'SREDNI_QRS']

        st.plotly_chart(fig_qrs_spocz, use_container_width=True)

    with prawy:
        wybrana_kolumna_spocz = f'QRS_{idx_segmentu_spocz + 1:02d}'
        y_values_spocz = df_qrs_spocz[wybrana_kolumna_spocz].values
        x_values_spocz = df_qrs_spocz.index

        fig_single_spocz = go.Figure()

        fig_single_spocz.add_trace(go.Scatter(
            x=list(x_values_spocz),
            y=list(y_values_spocz),
            mode='lines',
            line=dict(color=lekki_czerwony, width=4),
            name=wybrana_kolumna_spocz,
            fill='tozeroy',
            fillcolor='rgba(231, 76, 60, 0.2)'
        ))

        fig_single_spocz.update_layout(
            title=f"Analiza morfologii: {wybrana_kolumna_spocz}",
            xaxis_title="Próbki względem załamka R [n]",
            yaxis_title="Amplituda [mV]",
            template="plotly_dark",
            height=400,
            showlegend=False,
            shapes=[dict(
                type='line', yref='y', y0=0, y1=0,
                xref='x', x0=x_values_spocz.min(), x1=x_values_spocz.max(),
                line=dict(color="white", width=1, dash="dot")
            )]
        )

        st.plotly_chart(fig_single_spocz, use_container_width=True)


#%%-----------------------------SEKCJA 4 - Segmentacja QRS (WYSIŁKOWE)----------

st.markdown(f"""
    <hr style="margin-top: 10px;height:5px; border:none; color:#444444; background-color:#444444;" />
""", unsafe_allow_html=True)

st.markdown(f'<p style="margin-top: 0px; font-size: 18px; font-weight: bold; color:{zielony};">Segmentacja zespołu QRS – sygnał wysiłkowy</p>', unsafe_allow_html=True)

st.markdown(f"""
    <hr style="margin-top: 10px;height:5px; border:none; background-color:{zielony};" />
""", unsafe_allow_html=True)

col1, col2 = st.columns([4, 4.5])

with col1:
    col_left, col_right = st.columns([1, 1])

    with col_left:
        window_wys = st.slider("Szerokość okna [próbki]:", min_value=100, max_value=1000, value=250, step=10, key="window_wys")

        ecg_signal_wys = df_wys['ecg_filtrowany'].values
        qrs_dict_wys = {}

        for i, r in enumerate(peaks_wys):
            if r > window_wys and r + window_wys < len(ecg_signal_wys):
                segment = ecg_signal_wys[r - window_wys : r + window_wys].copy()
                segment = segment - np.mean(segment)
                qrs_dict_wys[f'QRS_{i+1:02d}'] = segment

        df_qrs_wys = pd.DataFrame(qrs_dict_wys, index=range(-window_wys, window_wys))

    with col_right:
        idx_segmentu_wys = st.slider("Wybierz numer zespołu QRS:", 0, len(peaks_wys)-1, 2, key="idx_wys")

    # Podgląd segmentu na tle całego sygnału
    r_center_wys = peaks_wys[idx_segmentu_wys]
    start_idx_wys = max(0, r_center_wys - window_wys)
    stop_idx_wys  = min(len(df_wys), r_center_wys + window_wys)

    fig_seg_wys = go.Figure()

    fig_seg_wys.add_trace(go.Scatter(
        x=df_wys['czas'],
        y=df_wys['ecg_filtrowany'],
        mode='lines',
        line=dict(color='rgba(150, 150, 150, 0.5)', width=1),
        name='Pełny sygnał'
    ))

    fig_seg_wys.add_trace(go.Scatter(
        x=df_wys['czas'].iloc[start_idx_wys:stop_idx_wys],
        y=df_wys['ecg_filtrowany'].iloc[start_idx_wys:stop_idx_wys],
        mode='lines',
        line=dict(color=zielony, width=3),
        name='Wybrany segment'
    ))

    fig_seg_wys.add_vrect(
        x0=df_wys['czas'].iloc[start_idx_wys],
        x1=df_wys['czas'].iloc[stop_idx_wys],
        fillcolor=zielony, opacity=0.2,
        layer="below", line_width=0,
    )

    fig_seg_wys.update_layout(
        title=f"Podgląd segmentu nr {idx_segmentu_wys + 1} (R w {df_wys['czas'].iloc[r_center_wys]:.2f} s)",
        xaxis_title="Czas [s]",
        yaxis_title="Amplituda",
        template="plotly_dark",
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    st.plotly_chart(fig_seg_wys, use_container_width=True)

with col2:
    lewy, prawy = st.columns([1, 1])

    with lewy:
        df_qrs_wys['SREDNI_QRS'] = df_qrs_wys.mean(axis=1)
        y_min_w = df_qrs_wys['SREDNI_QRS'].min()
        y_max_w = df_qrs_wys['SREDNI_QRS'].max()
        margines_w = (y_max_w - y_min_w) * 0.15

        fig_qrs_wys = px.line(df_qrs_wys,
                              labels={'index': 'Próbki względem R', 'value': 'Amplituda'},
                              title="Nałożone segmenty QRS z uśrednionym profilem")

        fig_qrs_wys.update_traces(line=dict(width=1, color='rgba(150, 150, 150, 0.3)'), opacity=0.4)
        fig_qrs_wys.update_layout(
            yaxis=dict(range=[y_min_w - margines_w, y_max_w + margines_w], fixedrange=False),
            template="plotly_dark",
            uirevision='constant'
        )

        fig_qrs_wys.for_each_trace(
            lambda trace: trace.update(line=dict(color=zielony, width=4), opacity=1)
            if trace.name == 'SREDNI_QRS' else ()
        )

        fig_qrs_wys.data = [t for t in fig_qrs_wys.data if t.name != 'SREDNI_QRS'] + \
                           [t for t in fig_qrs_wys.data if t.name == 'SREDNI_QRS']

        st.plotly_chart(fig_qrs_wys, use_container_width=True)

    with prawy:
        wybrana_kolumna_wys = f'QRS_{idx_segmentu_wys + 1:02d}'
        y_values_wys = df_qrs_wys[wybrana_kolumna_wys].values
        x_values_wys = df_qrs_wys.index

        fig_single_wys = go.Figure()

        fig_single_wys.add_trace(go.Scatter(
            x=list(x_values_wys),
            y=list(y_values_wys),
            mode='lines',
            line=dict(color=zielony, width=4),
            name=wybrana_kolumna_wys,
            fill='tozeroy',
            fillcolor='rgba(200, 255, 74, 0.2)'
        ))

        fig_single_wys.update_layout(
            title=f"Analiza morfologii: {wybrana_kolumna_wys}",
            xaxis_title="Próbki względem załamka R [n]",
            yaxis_title="Amplituda [mV]",
            template="plotly_dark",
            height=400,
            showlegend=False,
            shapes=[dict(
                type='line', yref='y', y0=0, y1=0,
                xref='x', x0=x_values_wys.min(), x1=x_values_wys.max(),
                line=dict(color="white", width=1, dash="dot")
            )]
        )

        st.plotly_chart(fig_single_wys, use_container_width=True)

#%%-----------------------------SEKCJA 5 - EMD (SPOCZYNKOWE)--------------------

import emd

st.markdown(f"""
    <hr style="margin-top: 10px;height:5px; border:none; color:#444444; background-color:#444444;" />
""", unsafe_allow_html=True)

st.markdown(f'<p style="margin-top: 0px; font-size: 18px; font-weight: bold; color:{lekki_czerwony};">Empirical Mode Decomposition – sygnał spoczynkowy</p>', unsafe_allow_html=True)

st.markdown(f"""
    <hr style="margin-top: 10px;height:5px; border:none; background-color:{lekki_czerwony};" />
""", unsafe_allow_html=True)

# ── Funkcja obliczająca IMF ────────────────────────────────────────────────────

@st.cache_data
def compute_imf(signal_array, metoda='CEEMDAN', n_ensembles=50, noise_scale=0.2):
    if metoda == 'EMD':
        return emd.sift.sift(signal_array)

    elif metoda == 'EEMD':
        try:
            return emd.sift.ensemble_sift(
                signal_array,
                nensembles=n_ensembles,
                ensemble_noise=noise_scale,
                nprocesses=1
            )
        except Exception:
            return emd.sift.sift(signal_array)

    elif metoda == 'CEEMDAN':
        try:
            return emd.sift.complete_ensemble_sift(
                signal_array,
                nensembles=n_ensembles,
                ensemble_noise=noise_scale,
                nprocesses=1
            )
        except IndexError:
            try:
                return emd.sift.ensemble_sift(
                    signal_array,
                    nensembles=n_ensembles,
                    ensemble_noise=noise_scale,
                    nprocesses=1
                )
            except Exception:
                return emd.sift.sift(signal_array)

# ── WIERSZ 1: wybór zakresu + metoda | wykres IMF ─────────────────────────────

row1_col1, row1_col2 = st.columns([1, 3])

with row1_col1:
    st.markdown(f"###### Zakres sygnału i metoda dekompozycji")

    signal_full = df_spocz['ecg'].to_numpy()
    czas_full   = df_spocz['czas'].to_numpy()

    min_czas_emd = float(czas_full.min())
    max_czas_emd = float(czas_full.max())

    zakres_emd = st.slider(
        "Wybierz zakres czasu [s]:",
        min_value=min_czas_emd,
        max_value=max_czas_emd,
        value=(min_czas_emd, min(min_czas_emd + 20.0, max_czas_emd)),
        step=0.5,
        key="emd_zakres_spocz"
    )

    mask = (czas_full >= zakres_emd[0]) & (czas_full <= zakres_emd[1])
    time_emd   = czas_full[mask]
    signal_emd = signal_full[mask]

    st.metric("Liczba próbek", f"{len(signal_emd)}")
    st.metric("Czas trwania",  f"{len(signal_emd)/fs:.1f} s")

    st.markdown("---")
    st.markdown("**Metoda dekompozycji:**")

    metoda_emd = st.selectbox(
        "Algorytm:",
        options=['CEEMDAN', 'EEMD', 'EMD'],
        index=0,
        key="metoda_emd_spocz",
        help="CEEMDAN > EEMD > EMD pod względem stabilności podziału IMF"
    )

    if metoda_emd in ['EEMD', 'CEEMDAN']:
        n_ensembles = st.slider(
            "Liczba realizacji (ensembles):",
            min_value=10, max_value=200, value=50, step=10,
            key="n_ens_spocz",
            help="Więcej = stabilniej, ale wolniej. 50 to dobry kompromis."
        )
        noise_scale = st.slider(
            "Poziom szumu:",
            min_value=0.05, max_value=1.0, value=0.2, step=0.05,
            key="noise_spocz",
            help="Zazwyczaj 0.1–0.4 działa dobrze dla EKG"
        )
    else:
        n_ensembles = 50
        noise_scale = 0.2

    st.markdown("---")

    # Podgląd wybranego fragmentu
    fig_preview = go.Figure()
    fig_preview.add_trace(go.Scatter(
        x=czas_full, y=signal_full,
        mode='lines', name='Pełny sygnał',
        line=dict(color=bialo_szary, width=1)
    ))
    fig_preview.add_trace(go.Scatter(
        x=time_emd, y=signal_emd,
        mode='lines', name='Wybrany fragment',
        line=dict(color=lekki_czerwony, width=2)
    ))
    fig_preview.add_vrect(
        x0=zakres_emd[0], x1=zakres_emd[1],
        fillcolor=lekki_czerwony, opacity=0.15,
        layer="below", line_width=0
    )
    fig_preview.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis_title="Czas [s]",
        yaxis_title="Amplituda"
    )
    with st.container(border=True):
        st.plotly_chart(fig_preview, use_container_width=True)

with row1_col2:
    st.markdown(f"###### Rozbicie sygnału na IMF")

    if len(signal_emd) < 100:
        st.warning("Wybierz dłuższy fragment sygnału (min. ~0.25 s).")
    else:
        with st.spinner(f"Trwa dekompozycja metodą {metoda_emd}..."):
            imf = compute_imf(
                signal_emd,
                metoda=metoda_emd,
                n_ensembles=n_ensembles,
                noise_scale=noise_scale
            )

        n_imfs = imf.shape[1]

        # Ostrzeżenie o fallbacku
        if metoda_emd in ['CEEMDAN', 'EEMD'] and n_imfs <= 3:
            st.warning(
                f"⚠️ {metoda_emd} napotkał błąd — użyto prostszej metody jako zapasowej. "
                "Spróbuj skrócić fragment sygnału lub zmniejszyć liczbę realizacji."
            )

        fig_imf = make_subplots(
            rows=n_imfs, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=[f"IMF {i+1}" for i in range(n_imfs)]
        )

        for i in range(n_imfs):
            fig_imf.add_trace(
                go.Scatter(
                    x=time_emd,
                    y=imf[:, i],
                    mode='lines',
                    name=f'IMF {i+1}',
                    line=dict(color=lekki_czerwony, width=1),
                    showlegend=False
                ),
                row=i+1, col=1
            )

        fig_imf.update_layout(
            height=120 * n_imfs,
            margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            template="plotly_dark",
        )

        with st.container(border=True):
            st.plotly_chart(fig_imf, use_container_width=True)

# ── WIERSZ 2: wybór IMF do rekonstrukcji | sygnał zrekonstruowany ─────────────

st.markdown(f"""
    <hr style="margin-top: 10px;height:3px; border:none; background-color:{lekki_szary};" />
""", unsafe_allow_html=True)

row2_col1, row2_col2 = st.columns([1, 3])

with row2_col1:
    st.markdown(f"###### Wybór IMF do rekonstrukcji")

    if len(signal_emd) >= 100:
        imf_options = [f"IMF {i+1}" for i in range(n_imfs)]

        wybrane_imf = st.multiselect(
            "Wybierz IMF do rekonstrukcji:",
            options=imf_options,
            default=imf_options[:3],
            key="imf_select_spocz"
        )

        selected_indices = [int(s.split()[1]) - 1 for s in wybrane_imf]

        if selected_indices:
            reconstructed = np.sum(imf[:, selected_indices], axis=1)
        else:
            reconstructed = np.zeros_like(signal_emd)
            st.info("Nie wybrano żadnego IMF — sygnał zrekonstruowany = 0.")

with row2_col2:
    st.markdown(f"###### Sygnał zrekonstruowany")

    if len(signal_emd) >= 100:
        if not selected_indices:
            st.info("Wybierz co najmniej jeden IMF w kolumnie obok.")
        else:
            fig_recon = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=[
                    "Sygnał oryginalny",
                    f"Rekonstrukcja: {', '.join(wybrane_imf)}"
                ]
            )

            fig_recon.add_trace(
                go.Scatter(
                    x=time_emd, y=signal_emd,
                    mode='lines', name='Oryginalny',
                    line=dict(color='rgba(174,174,174,0.7)', width=1),
                    showlegend=False
                ),
                row=1, col=1
            )

            fig_recon.add_trace(
                go.Scatter(
                    x=time_emd, y=reconstructed,
                    mode='lines', name='Zrekonstruowany',
                    line=dict(color=lekki_czerwony, width=2),
                    showlegend=False
                ),
                row=2, col=1
            )

            fig_recon.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                template="plotly_dark",
                xaxis2_title="Czas [s]",
            )

            with st.container(border=True):
                st.plotly_chart(fig_recon, use_container_width=True)
                
                
                
                
                
