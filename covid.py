# ==============================================================
# APPLICATION STREAMLIT – PROJET SDA (VERSION FINALE - CAHIER DES CHARGES)
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import os

# ----------------------------------------------------------------
# 0. CONFIGURATION DE LA PAGE
# ----------------------------------------------------------------
st.set_page_config(
    page_title="Projet Data - COVID-19",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------
# 1. SETUP NLTK & FONCTIONS
# ----------------------------------------------------------------
@st.cache_resource
def setup_nltk():
    try: nltk.data.find('tokenizers/punkt')
    except LookupError: nltk.download('punkt')
    try: nltk.data.find('corpora/stopwords')
    except LookupError: nltk.download('stopwords')

setup_nltk()

@st.cache_data
def load_data(path):
    # Chargement brut
    if not os.path.exists(path):
        return None, {}
    
    df_raw = pd.read_csv(path, low_memory=False)
    raw_shape = df_raw.shape
    
    # --- NETTOYAGE ---
    df = df_raw.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # 1. Suppression des lignes agrégats (Continents, Monde, Revenus...)
    # C'est ici qu'on passe de 400k à 395k lignes pour éviter les doublons statistiques
    df = df[~df['iso_code'].str.startswith('OWID')]
    df = df.dropna(subset=['iso_code'])

    # 2. Imputation des valeurs nulles pour les stats
    numeric_cols = ['new_cases', 'new_deaths', 'total_vaccinations_per_hundred', 'stringency_index']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 3. Variables dérivées pour l'analyse
    if 'population' in df.columns and 'new_cases' in df.columns:
        df['cases_per_million'] = (df['new_cases'] / df['population']) * 1000000
    else:
        df['cases_per_million'] = 0

    # Infos pour l'onglet Présentation
    data_info = {
        "raw_rows": raw_shape[0],
        "raw_cols": raw_shape[1],
        "clean_rows": df.shape[0],
        "clean_cols": df.shape[1],
        "missing": df.isna().sum(),
        "dtypes": df.dtypes
    }
    
    return df, data_info

# CHARGEMENT
DATA_PATH = "owid-covid-data.csv"
df, info = load_data(DATA_PATH)

if df is None:
    st.error(f"❌ Fichier '{DATA_PATH}' introuvable.")
    st.stop()

# ----------------------------------------------------------------
# 2. SIDEBAR - FILTRES DYNAMIQUES (Exigence : Filtres, Sliders...)
# ----------------------------------------------------------------
st.sidebar.header("🔍 Filtres Dynamiques")

# Filtre 1 : Période (Slider pour respecter le cahier des charges)
min_date = df['date'].min().date()
max_date = df['date'].max().date()

date_range = st.sidebar.slider(
    "1. Période d'analyse",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="DD/MM/YYYY" # Format français
)


# Filtre 2 : Pays (Multiselect)
all_countries = sorted(df['location'].unique())
selected_countries = st.sidebar.multiselect(
    "2. Choix des pays",
    options=all_countries,
    default=['France', 'United States', 'Germany', 'Brazil', 'China']
)

# Filtre 3 : Échelle Log (Checkbox)
use_log = st.sidebar.checkbox("3. Échelle Logarithmique (Graphiques)")

# Application des filtres
mask_date = (df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))
df_filtered = df.loc[mask_date]
df_countries = df_filtered[df_filtered['location'].isin(selected_countries)]

st.sidebar.markdown("---")
st.sidebar.info(f"Données affichées : {len(df_countries):,} lignes")

# ----------------------------------------------------------------
# 3. INTERFACE PRINCIPALE
# ----------------------------------------------------------------
st.title("🦠 Tableau de Bord Analytique COVID-19")
st.caption("Projet Data Visualization - Exploration complète du dataset OWID")

# Création des onglets selon le cahier des charges
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Présentation du Dataset",  # Exigence 1
    "🔢 Statistiques Descriptives", # Exigence 2
    "📈 Visualisations (Temps)",    # Exigence 3
    "🌍 Visualisations (Carte)",    # Exigence 3 suite
    "📰 Text Mining"               # Bonus
])

# ==============================================================
# ONGLET 1 : PRÉSENTATION (Répond point par point au cahier des charges)
# ==============================================================
with tab1:
    st.header("1. Présentation du Jeu de Données")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Source et Origine")
        st.markdown("""
        * **Source :** *Our World in Data* (Collaborateurs universitaires Oxford).
        * **URL :** [https://github.com/owid/covid-19-data](https://github.com/owid/covid-19-data)
        * **Contexte :** Données quotidiennes sur la pandémie (épidémiologie, vaccination, réponses politiques).
        """)
        
        st.subheader("📏 Dimensions")
        st.markdown(f"""
        * **Observations brutes :** `{info['raw_rows']:,}` lignes.
        * **Observations nettoyées :** `{info['clean_rows']:,}` lignes.
        * *Note : Les agrégats régionaux (ex: 'Europe', 'High income') ont été retirés pour l'analyse.*
        * **Nombre de variables :** `{info['clean_cols']}` colonnes.
        """)

    with col2:
        st.subheader("📖 Signification des Variables Clés")
        # Petit tableau explicatif manuel pour "Signification des variables"
        vars_def = pd.DataFrame([
            ["iso_code", "Code pays (ISO 3166-1 alpha-3)"],
            ["date", "Date de l'observation"],
            ["new_cases", "Nouveaux cas confirmés (quotidien)"],
            ["total_deaths", "Nombre cumulé de décès"],
            ["stringency_index", "Indice de sévérité des mesures (0-100)"],
            ["people_vaccinated_per_hundred", "% population avec au moins 1 dose"],
            ["gdp_per_capita", "PIB par habitant (USD)"]
        ], columns=["Variable", "Signification"])
        st.table(vars_def)

    st.subheader("⚠️ Valeurs Manquantes (Top 10)")
    # Calcul des manquants sur les colonnes principales uniquement pour la clarté
    missing_df = info['missing'].sort_values(ascending=False).head(10).to_frame(name="Nombre de NaN")
    st.dataframe(missing_df.T)

# ==============================================================
# ONGLET 2 : STATISTIQUES DESCRIPTIVES
# ==============================================================
with tab2:
    st.header("2. Statistiques Descriptives")
    st.markdown("Statistiques calculées sur la **période et les pays sélectionnés**.")
    
    if df_countries.empty:
        st.warning("Sélectionnez des pays pour voir les statistiques.")
    else:
        # Sélection de variables numériques pertinentes
        desc_vars = ['new_cases', 'new_deaths', 'stringency_index', 'people_vaccinated_per_hundred']
        # Vérif existence
        desc_vars = [v for v in desc_vars if v in df.columns]
        
        stats_df = df_countries[desc_vars].describe().T
        st.dataframe(stats_df.style.background_gradient(cmap="Blues"), use_container_width=True)

        st.subheader("Matrice de Corrélation")
        st.markdown("Analyse des liens entre les variables (sur l'ensemble filtré).")
        corr_matrix = df_countries[desc_vars].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title="Corrélation de Pearson")
        st.plotly_chart(fig_corr, use_container_width=True)

# ==============================================================
# ONGLET 3 : VISUALISATIONS TEMPORELLES (Min 5 graphiques -> Ici 4)
# ==============================================================
with tab3:
    st.header("3. Visualisations Temporelles")
    
    if df_countries.empty:
        st.warning("Aucun pays sélectionné.")
    else:
        # GRAPHIQUE 1 : Courbe des cas (Corrigé pour le Log)
        st.subheader("📈 1. Évolution des Cas Quotidiens")
        
        # Préparation des données pour éviter le bug du Log
        data_to_plot = df_countries.copy()
        if use_log:
            # On ne garde que les valeurs positives (>0) car log(0) est impossible
            data_to_plot = data_to_plot[data_to_plot['new_cases'] > 0]
            
        if not data_to_plot.empty:
            fig1 = px.line(
                data_to_plot, 
                x='date', 
                y='new_cases', 
                color='location', 
                log_y=use_log, # Utilise l'option cochée
                title=f"Nouveaux cas confirmés {'(Échelle Log)' if use_log else ''}"
            )
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("Aucune donnée positive à afficher en échelle logarithmique pour cette sélection.")
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            # GRAPHIQUE 2 : Aire Décès
            st.subheader("⚰️ 2. Décès Cumulés")
            fig2 = px.area(df_countries, x='date', y='total_deaths', color='location', 
                           title="Cumul des décès")
            st.plotly_chart(fig2, use_container_width=True)
        
        with col_g2:
            # GRAPHIQUE 3 : Stringency (Indice politique)
            st.subheader("👮 3. Indice de Rigueur")
            fig3 = px.line(df_countries, x='date', y='stringency_index', color='location',
                           title="Sévérité des mesures (0-100)")
            st.plotly_chart(fig3, use_container_width=True)

        # GRAPHIQUE 4 : Vaccination
        st.subheader("💉 4. Progression Vaccinale")
        if 'people_vaccinated_per_hundred' in df_countries.columns:
            fig4 = px.line(df_countries, x='date', y='people_vaccinated_per_hundred', color='location',
                           title="% Population vaccinée (au moins 1 dose)")
            st.plotly_chart(fig4, use_container_width=True)

# ==============================================================
# ONGLET 4 : ANALYSE GÉOGRAPHIQUE (CORRIGÉ)
# ==============================================================
with tab4:
    st.header("4. Analyse Géographique")
    st.markdown("**Note :** La carte affiche la valeur maximale connue pour chaque pays sur la période sélectionnée (pour éviter les données manquantes).")
    
    # 1. Choix de la métrique (Interaction utilisateur)
    metric_choice = st.selectbox(
        "Choisir l'indicateur à cartographier :",
        ["Total Cas / Million", "Total Décès / Million", "Taux Vaccination Complète", "PIB par Habitant"],
        index=2
    )
    
    # 2. Mapping choix -> colonne du DataFrame
    metric_map = {
        "Total Cas / Million": "total_cases_per_million",
        "Total Décès / Million": "total_deaths_per_million",
        "Taux Vaccination Complète": "people_fully_vaccinated_per_hundred",
        "PIB par Habitant": "gdp_per_capita"
    }
    col_map = metric_map[metric_choice]
    
    # 3. PRÉPARATION DES DONNÉES INTELLIGENTE
    # On filtre d'abord par la date limite
    last_date = pd.to_datetime(date_range[1])
    df_period_map = df[df['date'] <= last_date]
    
    # CORRECTION ICI : Au lieu de prendre la dernière ligne (.tail(1)), 
    # on prend le MAX (.max()) par pays pour la colonne choisie.
    # Cela permet de récupérer la dernière donnée de vaccination connue même si elle date d'il y a 3 jours.
    # On groupe par 'iso_code' ET 'location' pour garder le nom du pays.
    df_map = df_period_map.groupby(['iso_code', 'location'])[[col_map]].max().reset_index()
    
    # On retire les pays qui ont 0 ou NaN dans la colonne choisie pour ne pas fausser l'échelle
    df_map = df_map[df_map[col_map] > 0]

    # 4. GRAPHIQUE 5 : CARTE
    if not df_map.empty:
        fig_map = px.choropleth(
            df_map, 
            locations="iso_code", 
            color=col_map,
            hover_name="location", 
            # Utilisation de 'Reds' pour les cas/décès et 'Greens' ou 'Viridis' pour vaccins/PIB
            color_continuous_scale="Viridis" if "Vaccination" in metric_choice or "PIB" in metric_choice else "Reds",
            title=f"Carte Mondiale : {metric_choice}"
        )
        # Amélioration esthétique de la carte
        fig_map.update_geos(
            showframe=False, 
            showcoastlines=True, 
            projection_type="natural earth"
        )
        fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
        
        # 5. GRAPHIQUE 6 : TOP 10 BAR CHART
        st.subheader(f"📊 Top 10 Pays : {metric_choice}")
        top10 = df_map.nlargest(10, col_map).sort_values(col_map, ascending=True)
        fig_bar = px.bar(
            top10, 
            x=col_map, 
            y='location', 
            orientation='h', 
            text_auto='.2s',
            color=col_map, 
            title="Classement des pays les plus hauts"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
    else:
        st.warning("Pas de données disponibles pour cet indicateur sur la période choisie.")

# ==============================================================
# ONGLET 5 : TEXT MINING 
# ==============================================================
with tab5:
    st.header("5. Analyse Textuelle")
    
    txt_file = "article_medecine_sciences.txt"
    default_txt = """La pandémie de Covid-19 a marqué le XXIe siècle. 
    Les variants Delta et Omicron ont nécessité une adaptation constante des vaccins à ARN messager. 
    La politique sanitaire, entre confinement et pass vaccinal, a suscité de vifs débats en France et dans le monde."""
    
    if os.path.exists(txt_file):
        with open(txt_file, 'r', encoding='utf-8') as f: text = f.read()
    else:
        text = default_txt
        st.caption("Fichier externe non trouvé, utilisation d'un texte d'exemple.")

    # Tokenisation simple
    tokens = word_tokenize(text.lower(), language='french')
    stops = set(stopwords.words('french'))
    stops.update(['les', 'des', 'une', 'covid', '19', 'plus', 'cette'])
    clean_tokens = [w for w in tokens if w.isalpha() and w not in stops and len(w) > 2]
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Nuage de Mots")
        if clean_tokens:
            wc = WordCloud(background_color="white", width=800, height=400).generate(" ".join(clean_tokens))
            fig_wc, ax = plt.subplots()
            ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
            st.pyplot(fig_wc)
    
    with c2:
        st.subheader("Fréquences")
        if clean_tokens:
            df_freq = pd.DataFrame(Counter(clean_tokens).most_common(10), columns=['Mot', 'Freq'])
            st.dataframe(df_freq, hide_index=True)