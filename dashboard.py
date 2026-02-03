import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np
import warnings
from datetime import timedelta

warnings.filterwarnings("ignore")

# Configuration de la page
st.set_page_config(page_title="Smartcare Analytics", layout="wide", initial_sidebar_state="expanded")

# Styles CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .metric-box {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #30334e;
    }
    h1, h2, h3 {
        color: #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏥 Pilotage des Urgences & Prévisions")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data(mode_selection):
    if "Classique" in mode_selection:
        try:
            df = pd.read_csv("dataset_hebdo_2010-2016.csv")
            df['date'] = pd.to_datetime(df['date'])
            df = df.rename(columns={'date': 'Date', 'passages_urgences': 'Passages', 'Lits_Capacite': 'Lits', 'Indice_Tension': 'Tension'})
            return df.sort_values('Date')
        except FileNotFoundError:
            return pd.DataFrame()
    else:
        # Mode Crise (Stub pour l'exemple ou si le fichier existe)
        try:
            df = pd.read_csv("dataset_hebdo_psl_covid_2019_2021.csv")
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.rename(columns={'Passages_Hebdo': 'Passages', 'Lits_Capacite': 'Lits', 'Indice_Tension': 'Tension'})
            return df.sort_values('Date')
        except FileNotFoundError:
            return pd.DataFrame()

# Sidebar
st.sidebar.header("Paramètres")
mode = st.sidebar.radio("Contexte", ("Fonctionnement Classique (2010-2016)", "Gestion de Crise (2019-2021)"))

df = load_data(mode)

if not df.empty:
    # --- SECTION 1 : VUE HISTORIQUE & TENSION ---
    st.header("1. Analyse de l'Historique")
    
    # Filtres
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        years = sorted(df['Date'].dt.year.unique())
        selected_year = st.multiselect("Filtrer par année", years, default=years)
    
    if selected_year:
        df_filtered = df[df['Date'].dt.year.isin(selected_year)]
    else:
        df_filtered = df

    # KPI
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Patients", f"{df_filtered['Passages'].sum():,.0f}")
    with kpi2:
        st.metric("Moyenne Hebdo", f"{df_filtered['Passages'].mean():,.0f}")
    with kpi3:
        st.metric("Pic d'Affluence", f"{df_filtered['Passages'].max():,.0f}")
    with kpi4:
        tension_mean = df_filtered['Tension'].mean() if 'Tension' in df_filtered.columns else 0
        st.metric("Tension Moyenne", f"{tension_mean:.2f}")

    # Graphique Principal : Offre vs Demande
    fig_main = go.Figure()
    
    # Zone de capacité (Offre)
    fig_main.add_trace(go.Scatter(
        x=df_filtered['Date'], 
        y=df_filtered['Lits'],
        mode='lines',
        name='Capacité (Lits)',
        line=dict(color='#ef553b', width=1),
        fill='tozeroy',
        fillcolor='rgba(239, 85, 59, 0.1)'
    ))
    
    # Ligne des passages (Demande)
    fig_main.add_trace(go.Scatter(
        x=df_filtered['Date'], 
        y=df_filtered['Passages'],
        mode='lines+markers',
        name='Passages Urgences',
        line=dict(color='#00cc96', width=2),
        marker=dict(size=4)
    ))

    fig_main.update_layout(
        title="Confrontation Offre de Soins vs Demande Patients",
        xaxis_title="Semaines",
        yaxis_title="Volume",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig_main, use_container_width=True)

    # --- SECTION 2 : PRÉVISIONS 2017 ---
    st.divider()
    st.header("2. Prévisions Stratégiques (2017)")
    st.info("ℹ️ Cette section génère une **projection pour l'année 2017** basée sur la modélisation des cycles historiques (2010-2016).")

    # Bouton pour lancer le calcul (évite de ralentir le chargement initial)
    if st.checkbox("Afficher la projection 2017", value=False):
        
        with st.spinner("Modélisation SARIMA en cours (Saisonnalité 52 semaines)..."):
            # Préparation Time Series avec resample pour aligner les dates correctement
            ts_data = df.set_index('Date')['Passages'].resample('W-SUN').mean().ffill()
            
            # Modèle SARIMA (1,1,1)x(1,1,1,52) spécifiquement adapté aux données hebdo
            try:
                model = SARIMAX(ts_data, 
                                order=(1, 1, 1), 
                                seasonal_order=(1, 1, 1, 52),
                                enforce_stationarity=False, 
                                enforce_invertibility=False)
                model_fit = model.fit(disp=False)
                
                # Prévision 52 semaines (2017)
                forecast_steps = 52
                prediction = model_fit.get_forecast(steps=forecast_steps)
                pred_values = prediction.predicted_mean
                pred_ci = prediction.conf_int()
                
                # Création du Graphique de Projection
                fig_pred = go.Figure()

                # 1. Historique (2016 pour la continuité visuelle)
                last_year_data = ts_data[ts_data.index >= '2016-01-01']
                fig_pred.add_trace(go.Scatter(
                    x=last_year_data.index,
                    y=last_year_data.values,
                    mode='lines',
                    name='Historique (2016)',
                    line=dict(color='rgba(255, 255, 255, 0.3)', width=2)
                ))

                # 2. Prévision 2017 (Clairement distincte)
                fig_pred.add_trace(go.Scatter(
                    x=pred_values.index,
                    y=pred_values.values,
                    mode='lines+markers',
                    name='✨ Prévision 2017',
                    line=dict(color='#FFA15A', width=3, dash='dot'),
                    marker=dict(symbol='circle', size=6)
                ))

                # 3. Intervalle de Confiance
                fig_pred.add_trace(go.Scatter(
                    x=pred_values.index.tolist() + pred_values.index[::-1].tolist(),
                    y=pred_ci.iloc[:, 0].tolist() + pred_ci.iloc[:, 1][::-1].tolist(),
                    fill='toself',
                    fillcolor='rgba(255, 161, 90, 0.15)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name='Marge d\'erreur',
                    hoverinfo="skip"
                ))

                # Projection de la capacité (Lits) - Hypothèse constante ou dernière valeur connue
                last_capacity = df['Lits'].iloc[-1]
                capacity_line = [last_capacity] * len(pred_values)
                fig_pred.add_trace(go.Scatter(
                    x=pred_values.index,
                    y=capacity_line,
                    mode='lines',
                    name='Capacité Lits (Projetée)',
                    line=dict(color='#ef553b', width=2, dash='dash')
                ))

                fig_pred.update_layout(
                    title="Projection de l'Activité 2017",
                    xaxis_title="Semaines (2017)",
                    yaxis_title="Passages Estimés",
                    template="plotly_dark",
                    legend=dict(orientation="h", y=1.1)
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)

                # Analyse rapide
                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.success(f"**Moyenne prévisionnelle 2017** : {pred_values.mean():.0f} patients / semaine")
                with col_p2:
                    st.warning(f"**Pic critique estimé** : Semaine du {pred_values.idxmax().strftime('%d/%m')} avec {pred_values.max():.0f} patients")

            except Exception as e:
                st.error(f"Erreur de modélisation : {e}")

else:
    st.warning("Veuillez vérifier que les fichiers de données (CSV) sont bien présents dans le dossier.")
