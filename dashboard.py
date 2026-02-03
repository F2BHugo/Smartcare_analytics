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
    .recommendation-box {
        padding: 20px;
        background-color: #262730;
        border-left: 5px solid #00cc96;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .documentation-box {
        padding: 20px;
        background-color: #1e2130;
        border: 1px solid #30334e;
        border-radius: 5px;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏥 Pilotage des Urgences & Analyse Stratégique")

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data(mode_selection):
    if "Classique" in mode_selection:
        try:
            df = pd.read_csv("dataset_hebdo_2010-2016.csv")
            # Correction des noms de colonnes : 'Date', 'Passages_Hebdo'
            df['Date'] = pd.to_datetime(df['Date'])
            if 'Passages_Hebdo' in df.columns:
                df = df.rename(columns={'Passages_Hebdo': 'Passages'})
            elif 'passages_urgences' in df.columns:
                 df = df.rename(columns={'passages_urgences': 'Passages'})
            
            df = df.rename(columns={'Lits_Capacite': 'Lits', 'Indice_Tension': 'Tension'})
            return df.sort_values('Date')
        except FileNotFoundError:
            return pd.DataFrame()
    else:
        try:
            df = pd.read_csv("dataset_hebdo_psl_covid_2019_2021.csv")
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.rename(columns={'Passages_Hebdo': 'Passages', 'Lits_Capacite': 'Lits', 'Indice_Tension': 'Tension'})
            return df.sort_values('Date')
        except FileNotFoundError:
            return pd.DataFrame()

# Sidebar
st.sidebar.header("Paramètres Pilotage")
mode = st.sidebar.radio("Contexte", ("Fonctionnement Classique (2010-2016)", "Gestion de Crise (2019-2021)"))

df = load_data(mode)

# Création des onglets
tab_pilotage, tab_infographie, tab_doc = st.tabs(["🚀 Pilotage Opérationnel", "📊 Infographie & Décisions", "📚 Documentation"])

if not df.empty:
    
    # --- ONGLET 1 : PILOTAGE ---
    with tab_pilotage:
        st.header(f"Analyse Dynamique : {mode}")
        
        # Filtres
        years = sorted(df['Date'].dt.year.unique())
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
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

        # Graphique Principal
        st.subheader("Confrontation Offre de Soins vs Demande")
        
        high_flux_threshold = df_filtered['Passages'].quantile(0.90)
        
        fig_main = go.Figure()
        
        fig_main.add_trace(go.Scatter(
            x=df_filtered['Date'], 
            y=df_filtered['Lits'],
            mode='lines',
            name='Capacité (Lits)',
            line=dict(color='#ef553b', width=1),
            fill='tozeroy',
            fillcolor='rgba(239, 85, 59, 0.1)'
        ))
        
        fig_main.add_trace(go.Scatter(
            x=df_filtered['Date'], 
            y=df_filtered['Passages'],
            mode='lines+markers',
            name='Passages Urgences',
            line=dict(color='#00cc96', width=2),
            marker=dict(size=4)
        ))
        
        fig_main.add_hrect(
            y0=high_flux_threshold, y1=df_filtered['Passages'].max() * 1.1,
            fillcolor="rgba(255, 0, 0, 0.1)", layer="below", line_width=0,
            annotation_text="Flux Critique (>Top 10%)", annotation_position="top right"
        )

        fig_main.update_layout(
            xaxis_title="Semaines",
            yaxis_title="Volume Patients",
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_main, use_container_width=True)

        # Prévisions 2017
        st.divider()
        st.subheader("Prévisions Stratégiques (2017)")
        
        if st.checkbox("Lancer la simulation prévisionnelle SARIMA", value=False):
            with st.spinner("Modélisation SARIMA en cours (Saisonnalité 52 semaines)..."):
                try:
                    ts_data = df.set_index('Date')['Passages'].resample('W-SUN').mean().ffill()
                    
                    model = SARIMAX(ts_data, 
                                    order=(1, 1, 1), 
                                    seasonal_order=(1, 1, 1, 52),
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False)
                    model_fit = model.fit(disp=False)
                    
                    forecast_steps = 52
                    prediction = model_fit.get_forecast(steps=forecast_steps)
                    pred_values = prediction.predicted_mean
                    pred_ci = prediction.conf_int()
                    
                    fig_pred = go.Figure()

                    last_year_data = ts_data[ts_data.index >= '2016-01-01']
                    fig_pred.add_trace(go.Scatter(
                        x=last_year_data.index, y=last_year_data.values,
                        mode='lines', name='Historique (2016)',
                        line=dict(color='rgba(255, 255, 255, 0.3)', width=2)
                    ))

                    fig_pred.add_trace(go.Scatter(
                        x=pred_values.index, y=pred_values.values,
                        mode='lines+markers', name='✨ Prévision 2017',
                        line=dict(color='#FFA15A', width=3, dash='dot'),
                        marker=dict(symbol='circle', size=6)
                    ))

                    fig_pred.add_trace(go.Scatter(
                        x=pred_values.index.tolist() + pred_values.index[::-1].tolist(),
                        y=pred_ci.iloc[:, 0].tolist() + pred_ci.iloc[:, 1][::-1].tolist(),
                        fill='toself', fillcolor='rgba(255, 161, 90, 0.15)',
                        line=dict(color='rgba(0,0,0,0)'), name="Marge d'erreur", hoverinfo="skip"
                    ))

                    last_capacity = df['Lits'].iloc[-1]
                    fig_pred.add_trace(go.Scatter(
                        x=pred_values.index, y=[last_capacity]*52,
                        mode='lines', name='Capacité Lits',
                        line=dict(color='#ef553b', width=2, dash='dash')
                    ))

                    fig_pred.update_layout(
                        title="Projection de l'Activité 2017",
                        xaxis_title="Semaines (2017)", yaxis_title="Passages Estimés",
                        template="plotly_dark", legend=dict(orientation="h", y=1.1)
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Erreur de modélisation : {e}")

    # --- ONGLET 2 : INFOGRAPHIE ---
    with tab_infographie:
        st.header("📊 Infographie de Synthèse : Analyse & Décisions")
        st.markdown("Ce rapport synthétise l'état des lieux de PSL-CFX, la saisonnalité et l'impact potentiel d'une crise sanitaire.")
        
        try:
            df_classique = pd.read_csv("dataset_hebdo_2010-2016.csv")
            df_classique['Date'] = pd.to_datetime(df_classique['Date']) 
            if 'Passages_Hebdo' in df_classique.columns:
                 df_classique = df_classique.rename(columns={'Passages_Hebdo': 'Passages'})
            elif 'passages_urgences' in df_classique.columns:
                 df_classique = df_classique.rename(columns={'passages_urgences': 'Passages'})
            
            df_classique = df_classique.rename(columns={'Lits_Capacite': 'Lits'})
            
            df_crise = pd.read_csv("dataset_hebdo_psl_covid_2019_2021.csv")
            df_crise['Date'] = pd.to_datetime(df_crise['Date'])
            df_crise = df_crise.rename(columns={'Passages_Hebdo': 'Passages', 'Lits_Capacite': 'Lits'})
        except Exception as e:
            st.error(f"Erreur de chargement des données pour l'infographie : {e}")
            st.stop()

        col_inf1, col_inf2, col_inf3 = st.columns(3)
        avg_passages = df_classique['Passages'].mean()
        avg_lits = df_classique['Lits'].mean()
        occupancy_rate = (avg_passages / avg_lits) * 100
        
        col_inf1.metric("Capacité d'Accueil Moyenne", f"{avg_lits:.0f} Lits")
        col_inf2.metric("Demande Moyenne (Patients)", f"{avg_passages:.0f} / sem")
        col_inf3.metric("Taux d'Occupation Théorique", f"{occupancy_rate:.1f}%")
        st.progress(min(occupancy_rate/100, 1.0))
        st.caption("Barre de charge du service (Moyenne Historique)")

        st.divider()

        col_saison1, col_saison2 = st.columns([2, 1])
        with col_saison1:
            df_classique['Mois'] = df_classique['Date'].dt.month_name()
            df_classique['Mois_Num'] = df_classique['Date'].dt.month
            seasonality = df_classique.groupby(['Mois_Num', 'Mois'])['Passages'].mean().reset_index().sort_values('Mois_Num')
            fig_season = px.bar(seasonality, x='Mois', y='Passages', 
                                title="Affluence Moyenne par Mois", text_auto='.0f',
                                color='Passages', color_continuous_scale='OrRd')
            fig_season.update_layout(template="plotly_dark", xaxis_title=None, yaxis_title="Passages Moyens")
            st.plotly_chart(fig_season, use_container_width=True)
            
        with col_saison2:
            st.markdown("#### 💡 Analyse")
            try:
                peak_month = seasonality.loc[seasonality['Passages'].idxmax(), 'Mois']
                low_month = seasonality.loc[seasonality['Passages'].idxmin(), 'Mois']
                st.write(f"- **Mois le plus chargé** : {peak_month}")
                st.write(f"- **Mois le plus calme** : {low_month}")
                st.write("- On observe une cyclicité marquée nécessitant une modulation des effectifs.")
            except:
                st.write("Données insuffisantes.")

        st.divider()

        crisis_avg = df_crise['Passages'].mean()
        crisis_max = df_crise['Passages'].max()
        delta_avg = ((crisis_avg - avg_passages) / avg_passages) * 100
        delta_max = ((crisis_max - df_classique['Passages'].max()) / df_classique['Passages'].max()) * 100
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("Impact sur la Moyenne", f"{delta_avg:+.1f}%", help="Variation par rapport à la normale")
        with col_c2:
            st.metric("Impact sur les Pics", f"{delta_max:+.1f}%", delta_color="inverse", help="Augmentation de la charge maximale")
            
        fig_impact = go.Figure()
        fig_impact.add_trace(go.Box(y=df_classique['Passages'], name='Situation Normale', marker_color='#00cc96'))
        fig_impact.add_trace(go.Box(y=df_crise['Passages'], name='Situation Crise', marker_color='#ef553b'))
        fig_impact.update_layout(title="Comparaison des Distributions : Normal vs Crise", template="plotly_dark")
        st.plotly_chart(fig_impact, use_container_width=True)

        st.divider()
        st.subheader("4. Recommandations Stratégiques")
        rec_html = """
        <div class="recommendation-box">
            <h4>✅ 1. Gestion des Pics Saisonniers</h4>
            <p>Le flux de patients est maximal en hiver. Il est recommandé d'augmenter la capacité d'accueil de <b>15%</b> durant les mois de Janvier et Février.</p>
        </div>
        <div class="recommendation-box">
            <h4>✅ 2. Préparation aux Crises</h4>
            <p>En cas de crise sanitaire, l'hôpital doit pouvoir absorber une surcharge de <b>+30%</b>. Un plan de déploiement rapide de lits supplémentaires est requis.</p>
        </div>
        <div class="recommendation-box">
            <h4>✅ 3. Optimisation du Parcours</h4>
            <p>Pour réduire la tension moyenne, il est suggéré de renforcer le tri à l'entrée et de développer la télémédecine pour les cas non critiques.</p>
        </div>
        """
        st.markdown(rec_html, unsafe_allow_html=True)

    # --- ONGLET 3 : DOCUMENTATION (Nouveau) ---
    with tab_doc:
        st.header("📚 Documentation du Dashboard")
        
        st.markdown("""
        ### Objectif
        Ce tableau de bord a été conçu pour permettre aux décideurs hospitaliers de suivre l'activité des urgences de PSL-CFX, d'analyser les cycles historiques et de prévisualiser l'impact potentiel de crises sanitaires.

        ### Sources de Données
        *   **`dataset_hebdo_2010-2016.csv`** : Historique des passages aux urgences et capacité en lits en fonctionnement normal.
        *   **`dataset_hebdo_psl_covid_2019_2021.csv`** : Données spécifiques collectées durant la crise COVID-19 pour simuler des scénarios de surcharge.

        ### Méthodologie
        *   **Alignement Temporel** : Les données sont agrégées à la semaine (Hebdo).
        *   **Modélisation Prédictive (SARIMA)** : 
            *   Nous utilisons un modèle SARIMA (Seasonal AutoRegressive Integrated Moving Average) pour projeter l'activité future.
            *   **Paramètres** : `(1, 1, 1)` pour la partie non-saisonnière et `(1, 1, 1, 52)` pour la saisonnalité (cycle annuel de 52 semaines).
            *   Pour éviter les artefacts de calcul (valeurs nulles), un rééchantillonnage intelligent (`resample('W-SUN')`) est appliqué avant l'entraînement du modèle.
        *   **Calcul des KPI** :
            *   *Taux d'Occupation Théorique* = (Moyenne Passages / Moyenne Lits) * 100.
            *   *Flux Critique* : Défini comme le top 10% des semaines les plus chargées historiquement.

        ### Guide d'Utilisation
        1.  **Onglet "Pilotage Opérationnel"** :
            *   Utilisez la barre latérale pour basculer entre le mode "Classique" et "Crise".
            *   Filtrez par année pour zoomer sur des périodes spécifiques.
            *   Activez la case à cocher "Lancer la simulation prévisionnelle" pour voir les projections 2017.
        2.  **Onglet "Infographie & Décisions"** :
            *   Consultez cette vue pour un résumé exécutif prêt à être partagé.
            *   Analysez les graphiques de saisonnalité pour planifier les congés et les renforts.
        
        ### Glossaire
        *   **SARIMA** : Modèle statistique utilisé pour prédire des séries temporelles avec une composante saisonnière forte.
        *   **Indice de Tension** : Indicateur composite reflétant la pression sur le service (ratio passages/lits pondéré).
        """)
        
        st.info("ℹ️ Pour toute question technique ou demande d'évolution, veuillez contacter l'équipe Data Science.")

else:
    st.warning("Veuillez vérifier que les fichiers de données (CSV) sont bien présents dans le dossier.")
