# 📊 Projet Data Management & Visualisation - DU SDA 2025/2026

## Analyse du Dataset OWID COVID-19

**Thématique :** Santé publique & épidémiologie  
**Source :** Our World in Data (OWID)  
**Volume :** &gt;200 000 lignes

---

### 👥 Équipe
- Ibrahima Bâ
- Mahamat Sultan  
- Moustapha Mendy

---

### 🎯 Objectifs
- Explorer, nettoyer et enrichir un dataset de grande dimension
- Créer des visualisations pertinentes
- Développer une application Streamlit avec analyse textuelle

---

### 🛠️ Stack Technique
- **Python** : Pandas, NumPy
- **Visualisation** : Matplotlib, Seaborn
- **App** : Streamlit (fichier `covid.py`)

---

### 📋 Pipeline de Données

| Étape | Action | Détails |
|-------|--------|---------|
| **1. EDA** | Analyse exploratoire | Structure, types, valeurs manquantes |
| **2. Nettoyage** | Suppression agrégats OWID | Filtre `iso_code` ne commençant pas par "OWID" |
| **3. Imputation** | Gestion des NaN | `new_cases`, `new_deaths` → 0 |
| **4. Feature Engineering** | 2 nouvelles variables | `cases_per_million`, `policy_level` |

---

### 🔧 Feature Engineering

| Variable | Description | Méthode |
|----------|-------------|---------|
| `cases_per_million` | Normalisation des cas par population | `(new_cases / population) * 1M` |
| `policy_level` | Catégorisation de la rigueur des politiques | `stringency_index` → Basse/Modérée/Élevée |

---

### 📈 Visualisations Clés
- Distribution des niveaux de rigueur politique
- Matrice de corrélation (cas, décès, vaccination, restrictions)

---

### 🚀 Lancement

```bash
# Data Management
jupyter notebook analyse_covid.ipynb

# Application Streamlit

📦 Livrables

✅ Notebook d'analyse et nettoyage
✅ Script covid.py (reproductibilité)
✅ Application Streamlit interactive



streamlit run covid.py
