"""
Health-InsurTech : Prédicteur de frais de santé
Application Streamlit complète - RGPD Compliant
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import logging
import hashlib

# ─── Configuration de la page ────────────────────────────────────────────────
st.set_page_config(
    page_title="Health-InsurTech",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Custom CSS (Design & Accessibilité) ──────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
    .stApp { background: linear-gradient(135deg, #e8f4f8 0%, #f0f4f8 100%); }
    .rgpd-banner {
        background: linear-gradient(90deg, #1a365d, #2b6cb0);
        color: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: white; border-radius: 16px; padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08); border-left: 5px solid #2b6cb0;
        text-align: center;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1a365d; }
    .section-header {
        font-size: 1.5rem; font-weight: 700; color: #1a365d;
        border-bottom: 3px solid #2b6cb0; padding-bottom: 0.5rem; margin: 2rem 0 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #1a365d, #2b6cb0);
        color: white; padding: 2rem; border-radius: 20px; text-align: center;
    }
    .bias-warning { background: #fff5f5; border: 2px solid #fc8181; padding: 1rem; border-radius: 10px; }
    .bias-ok { background: #f0fff4; border: 2px solid #68d391; padding: 1rem; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─── Authentification ─────────────────────────────────────────────────────────
USERS = {"admin": hashlib.sha256("Admin2025!".encode()).hexdigest(), "demo": hashlib.sha256("Demo2025!".encode()).hexdigest()}

def check_password():
    if "authenticated" not in st.session_state: st.session_state.authenticated = False
    if st.session_state.authenticated: return True
    
    st.markdown("## 🏥 Health-InsurTech — Connexion")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        u = st.text_input("Identifiant")
        p = st.text_input("Mot de passe", type="password")
        if st.button("Se connecter"):
            if u in USERS and USERS[u] == hashlib.sha256(p.encode()).hexdigest():
                st.session_state.authenticated = True
                st.session_state.username = u
                st.rerun()
            else: st.error("Identifiant ou mot de passe incorrect")
    return False

# ─── Consentement RGPD ───────────────────────────────────────────────────────
def show_rgpd_consent():
    if "rgpd_accepted" not in st.session_state: st.session_state.rgpd_accepted = False
    if not st.session_state.rgpd_accepted:
        st.markdown('<div class="rgpd-banner"><h3>🔒 Protection de vos données</h3><p>Nous utilisons vos données uniquement pour la simulation. Aucune donnée n\'est stockée.</p></div>', unsafe_allow_html=True)
        if st.button("✅ J'accepte"): 
            st.session_state.rgpd_accepted = True
            st.rerun()
        st.stop()

# ─── Chargement & Modèle ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("insurance_data.csv")
    # Nettoyage minimal
    cols_to_fix = ["age", "bmi", "children", "charges"]
    for c in cols_to_fix: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols_to_fix + ["smoker", "sex"])
    
    # Anonymisation (PII removal)
    pii = ["nom", "prenom", "email", "telephone", "numero_secu_sociale", "adresse_ip"]
    df_anon = df.drop(columns=[c for c in pii if c in df.columns])
    return df, df_anon

@st.cache_resource
def train_model(df):
    le_smoker, le_sex, le_region = LabelEncoder(), LabelEncoder(), LabelEncoder()
    
    X = df[["age", "bmi", "children", "smoker", "sex", "region"]].copy()
    X["smoker_enc"] = le_smoker.fit_transform(X["smoker"])
    X["sex_enc"] = le_sex.fit_transform(X["sex"])
    X["region_enc"] = le_region.fit_transform(X["region"])
    
    X_m = X[["age", "bmi", "children", "smoker_enc", "sex_enc", "region_enc"]]
    y = df["charges"]
    
    X_train, X_test, y_train, y_test = train_test_split(X_m, y, test_size=0.2, random_state=42)
    
    lr = LinearRegression().fit(X_train, y_train)
    dt = DecisionTreeRegressor(max_depth=4).fit(X_train, y_train)
    
    metrics = {
        "lr": {"r2": r2_score(y_test, lr.predict(X_test)), "mae": mean_absolute_error(y_test, lr.predict(X_test)), "coefs": lr.coef_},
        "dt": {"r2": r2_score(y_test, dt.predict(X_test)), "mae": mean_absolute_error(y_test, dt.predict(X_test))}
    }
    return lr, dt, le_smoker, le_sex, le_region, metrics, X_test, y_test

# ─── Pages ────────────────────────────────────────────────────────────────────
def page_dashboard(df_anon, metrics):
    st.markdown('<div class="section-header">📊 Dashboard</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Clients", len(df_anon))
    c2.metric("Frais Moyens", f"{df_anon['charges'].mean():,.0f}€")
    c3.metric("Précision Modèle (R²)", f"{metrics['lr']['r2']:.2f}")

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.scatter(df_anon, x="age", y="charges", color="smoker", title="Âge vs Frais")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        reg_col = "region_fr" if "region_fr" in df_anon.columns else "region"
        fig2 = px.box(df_anon, x=reg_col, y="charges", title="Répartition par Région")
        st.plotly_chart(fig2, use_container_width=True)

def page_simulator(lr, le_sex, le_region):
    st.markdown('<div class="section-header">🔮 Simulateur</div>', unsafe_allow_html=True)
    with st.form("sim"):
        age = st.slider("Âge", 18, 90, 30)
        bmi = st.number_input("IMC", 15.0, 50.0, 25.0)
        children = st.number_input("Enfants", 0, 10, 0)
        smoker = st.radio("Fumeur ?", ["Non", "Oui"])
        sex = st.radio("Sexe", ["Homme", "Femme"])
        region = st.selectbox("Région", le_region.classes_)
        
        if st.form_submit_button("Calculer"):
            s_enc = 1 if smoker == "Oui" else 0
            x_enc = le_sex.transform(["male" if sex == "Homme" else "female"])[0]
            r_enc = le_region.transform([region])[0]
            
            res = lr.predict([[age, bmi, children, s_enc, x_enc, r_enc]])[0]
            st.markdown(f'<div class="prediction-box"><h2>Estimation : {max(0, res):,.2f} €</h2></div>', unsafe_allow_html=True)

def page_model_bias(df, lr, le_smoker, le_sex, le_region):
    st.markdown('<div class="section-header">🧮 Biais & Interprétabilité</div>', unsafe_allow_html=True)
    # Analyse simplifiée des biais
    df_bias = df.copy()
    X = df[["age", "bmi", "children", "smoker", "sex", "region"]].copy()
    X["smoker_enc"] = le_smoker.transform(X["smoker"])
    X["sex_enc"] = le_sex.transform(X["sex"])
    X["region_enc"] = le_region.transform(X["region"])
    df_bias["pred"] = lr.predict(X[["age", "bmi", "children", "smoker_enc", "sex_enc", "region_enc"]])
    df_bias["erreur"] = df_bias["charges"] - df_bias["pred"]
    
    st.write("### Erreur moyenne par catégorie")
    st.dataframe(df_bias.groupby("smoker")["erreur"].mean())
    
    fig = px.bar(x=["Âge", "IMC", "Enfants", "Fumeur", "Sexe", "Région"], y=lr.coef_, title="Poids des variables")
    st.plotly_chart(fig)

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    if not check_password(): return
    show_rgpd_consent()
    
    try:
        df, df_anon = load_data()
        lr, dt, le_smoker, le_sex, le_region, metrics, X_test, y_test = train_model(df)
    except FileNotFoundError:
        st.error("Fichier insurance_data.csv introuvable !")
        return

    page = st.sidebar.radio("Navigation", ["Dashboard", "Simulateur", "Biais & Modèle"])
    
    if page == "Dashboard": page_dashboard(df_anon, metrics)
    elif page == "Simulateur": page_simulator(lr, le_sex, le_region)
    elif page == "Biais & Modèle": page_model_bias(df, lr, le_smoker, le_sex, le_region)

if __name__ == "__main__":
    main()