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
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import logging
import hashlib
import datetime
import os
import json

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Health-InsurTech",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }
    .main { background-color: #f0f4f8; }
    .stApp { background: linear-gradient(135deg, #e8f4f8 0%, #f0f4f8 100%); }

    /* RGPD Banner */
    .rgpd-banner {
        background: linear-gradient(90deg, #1a365d, #2b6cb0);
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .rgpd-banner strong { color: #90cdf4; }

    /* KPI Cards */
    .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 5px solid #2b6cb0;
        text-align: center;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        font-family: 'JetBrains Mono', monospace;
    }
    .kpi-label { color: #718096; font-size: 0.85rem; margin-top: 0.3rem; }

    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1a365d;
        border-bottom: 3px solid #2b6cb0;
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }

    /* Prediction Result */
    .prediction-box {
        background: linear-gradient(135deg, #1a365d, #2b6cb0);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 8px 30px rgba(43,108,176,0.4);
    }
    .prediction-amount {
        font-size: 3.5rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        color: #90cdf4;
    }
    .prediction-label { font-size: 1rem; color: #e2e8f0; margin-top: 0.5rem; }

    /* Warning / bias cards */
    .bias-warning {
        background: #fff5f5;
        border: 2px solid #fc8181;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }
    .bias-ok {
        background: #f0fff4;
        border: 2px solid #68d391;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
    }

    /* WCAG accessibility */
    .stButton > button {
        background: #2b6cb0;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1.05rem;
        font-weight: 600;
        font-family: 'Sora', sans-serif;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #1a365d;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(43,108,176,0.4);
    }
    .stButton > button:focus {
        outline: 3px solid #90cdf4;
        outline-offset: 2px;
    }

    /* Skip to content link (WCAG) */
    .skip-link {
        position: absolute;
        top: -40px;
        left: 0;
        background: #2b6cb0;
        color: white;
        padding: 8px;
        text-decoration: none;
        z-index: 1000;
    }
    .skip-link:focus { top: 0; }

    /* High contrast labels */
    label { color: #1a365d !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Authentication ────────────────────────────────────────────────────────────
USERS = {
    "admin": hashlib.sha256("Admin2025!".encode()).hexdigest(),
    "demo":  hashlib.sha256("Demo2025!".encode()).hexdigest(),
}

def check_password():
    """Simple authentication gate."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    st.markdown("## 🏥 Health-InsurTech — Connexion")
    st.markdown("*Veuillez vous authentifier pour accéder à l'application.*")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Identifiant", key="login_user", autocomplete="username",
                                  help="Saisissez votre identifiant")
        password = st.text_input("Mot de passe", type="password", key="login_pass",
                                  autocomplete="current-password",
                                  help="Saisissez votre mot de passe")
        if st.button("Se connecter", key="login_btn"):
            h = hashlib.sha256(password.encode()).hexdigest()
            if username in USERS and USERS[username] == h:
                st.session_state.authenticated = True
                st.session_state.username = username
                logger.info(f"LOGIN SUCCESS | user={username} | ip=session")
                st.rerun()
            else:
                logger.warning(f"LOGIN FAILED | user={username}")
                st.error("❌ Identifiant ou mot de passe incorrect.")
        st.caption("Démo : `demo` / `Demo2025!`")
    return False


# ─── RGPD Consent ─────────────────────────────────────────────────────────────
def show_rgpd_consent():
    if "rgpd_accepted" not in st.session_state:
        st.session_state.rgpd_accepted = False

    if not st.session_state.rgpd_accepted:
        st.markdown("""
        <div class="rgpd-banner" role="dialog" aria-modal="true" aria-labelledby="rgpd-title">
            <h2 id="rgpd-title" style="margin:0 0 0.5rem 0;">🔒 Politique de confidentialité & RGPD</h2>
            <p>Cette application collecte des données de santé (<strong>IMC, âge, statut fumeur</strong>)
            uniquement dans le but d'estimer vos frais médicaux. Ces données ne sont <strong>jamais
            stockées, transmises ou vendues</strong> à des tiers. Elles sont traitées en mémoire
            le temps de votre session et supprimées à la déconnexion.</p>
            <p>Conformément au <strong>RGPD (Règlement UE 2016/679)</strong>, vous disposez d'un droit
            d'accès, de rectification et d'opposition. Pour exercer vos droits : 
            <strong>dpo@healthinsurtech.fr</strong></p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ J'accepte la politique de confidentialité", key="rgpd_accept"):
                st.session_state.rgpd_accepted = True
                logger.info("RGPD_CONSENT | accepted=True")
                st.rerun()
        with col2:
            if st.button("❌ Refuser et quitter", key="rgpd_refuse"):
                logger.info("RGPD_CONSENT | accepted=False")
                st.warning("Vous avez refusé. L'application ne peut pas fonctionner sans votre consentement.")
                st.stop()
        st.stop()


# ─── Data Loading & Model ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and clean data — strip PII columns for display."""
    df = pd.read_csv("insurance_data.csv")
    df["age"]      = pd.to_numeric(df["age"], errors="coerce")
    df["bmi"]      = pd.to_numeric(df["bmi"], errors="coerce")
    df["children"] = pd.to_numeric(df["children"], errors="coerce")
    df["charges"]  = pd.to_numeric(df["charges"], errors="coerce")
    df = df.dropna(subset=["age", "bmi", "children", "charges", "smoker", "sex"])

    # Anonymized view (PII removed)
    pii_cols = ["nom", "prenom", "email", "telephone", "numero_secu_sociale",
                "adresse_ip", "date_naissance", "ville", "code_postal"]
    df_anon = df.drop(columns=[c for c in pii_cols if c in df.columns])
    return df, df_anon


@st.cache_resource
def train_model(df):
    """Train interpretable models and return results."""
    le_smoker = LabelEncoder()
    le_sex    = LabelEncoder()
    le_region = LabelEncoder()

    X = df[["age", "bmi", "children", "smoker", "sex", "region"]].copy()
    X["smoker_enc"] = le_smoker.fit_transform(X["smoker"])
    X["sex_enc"]    = le_sex.fit_transform(X["sex"])
    X["region_enc"] = le_region.fit_transform(X["region"])
    X_model = X[["age", "bmi", "children", "smoker_enc", "sex_enc", "region_enc"]]
    y = df["charges"]

    X_train, X_test, y_train, y_test = train_test_split(X_model, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Decision Tree (max_depth=4 for interpretability)
    dt = DecisionTreeRegressor(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)

    metrics = {
        "lr": {
            "mae":  mean_absolute_error(y_test, y_pred_lr),
            "r2":   r2_score(y_test, y_pred_lr),
            "coefs": dict(zip(
                ["Âge", "IMC", "Enfants", "Fumeur", "Sexe", "Région"],
                lr.coef_
            )),
            "intercept": lr.intercept_,
        },
        "dt": {
            "mae": mean_absolute_error(y_test, y_pred_dt),
            "r2":  r2_score(y_test, y_pred_dt),
        }
    }
    return lr, dt, le_smoker, le_sex, le_region, metrics, X_test, y_test, y_pred_lr, y_pred_dt


# ─── Bias Analysis ────────────────────────────────────────────────────────────
def analyze_bias(df, model, le_smoker, le_sex, le_region):
    """Check if model over-penalises certain groups."""
    X = df[["age", "bmi", "children", "smoker", "sex", "region"]].copy()
    X["smoker_enc"] = le_smoker.transform(X["smoker"])
    X["sex_enc"]    = le_sex.transform(X["sex"])
    X["region_enc"] = le_region.transform(X["region"])
    X_m = X[["age", "bmi", "children", "smoker_enc", "sex_enc", "region_enc"]]
    df = df.copy()
    df["predicted"] = model.predict(X_m)
    df["error"]     = df["charges"] - df["predicted"]

    bias = {}
    for grp in ["smoker", "sex", "region"]:
        bias[grp] = df.groupby(grp)[["charges", "predicted", "error"]].mean().round(2)
    return bias, df


# ─── Sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("### 🏥 Health-InsurTech")
        st.markdown(f"*Connecté : {st.session_state.get('username','?')}*")
        st.divider()

        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "🔮 Simulateur", "🧮 Modèle & Biais", "📋 RGPD & Accès"],
            key="nav",
            help="Choisissez une section de l'application"
        )
        st.divider()
        st.caption("v1.0 — RGPD Compliant")
        st.caption("© 2025 Health-InsurTech")

        if st.button("🚪 Déconnexion", key="logout"):
            logger.info(f"LOGOUT | user={st.session_state.get('username','?')}")
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    return page


# ─── Page: Dashboard ──────────────────────────────────────────────────────────
def page_dashboard(df_anon, metrics):
    st.markdown('<div class="section-header">📊 Tableau de Bord — Aperçu des Données</div>', unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="kpi-card" role="region" aria-label="Nombre de clients">
            <div class="kpi-value">{len(df_anon):,}</div>
            <div class="kpi-label">Clients analysés</div></div>""", unsafe_allow_html=True)
    with c2:
        avg = df_anon["charges"].mean()
        st.markdown(f"""<div class="kpi-card" role="region" aria-label="Frais moyens">
            <div class="kpi-value">{avg:,.0f}€</div>
            <div class="kpi-label">Frais moyens annuels</div></div>""", unsafe_allow_html=True)
    with c3:
        pct_smoker = (df_anon["smoker"] == "yes").mean() * 100
        st.markdown(f"""<div class="kpi-card" role="region" aria-label="Pourcentage fumeurs">
            <div class="kpi-value">{pct_smoker:.1f}%</div>
            <div class="kpi-label">Fumeurs</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="kpi-card" role="region" aria-label="R² du modèle">
            <div class="kpi-value">{metrics['lr']['r2']:.2f}</div>
            <div class="kpi-label">R² Modèle (Régression)</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    # Scatter: IMC vs Charges coloré par âge
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔵 IMC × Frais médicaux (par âge)")
        fig = px.scatter(
            df_anon, x="bmi", y="charges", color="age",
            color_continuous_scale="Blues",
            labels={"bmi": "IMC (kg/m²)", "charges": "Frais (€)", "age": "Âge"},
            opacity=0.65, height=400,
            hover_data={"children": True, "smoker": True}
        )
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="#f8fafc",
                          font_family="Sora")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🔴 Âge × Frais (fumeurs vs non-fumeurs)")
        fig2 = px.scatter(
            df_anon, x="age", y="charges", color="smoker",
            color_discrete_map={"yes": "#E53E3E", "no": "#2B6CB0"},
            labels={"age": "Âge", "charges": "Frais (€)", "smoker": "Fumeur"},
            opacity=0.65, height=400,
        )
        fig2.update_layout(paper_bgcolor="white", plot_bgcolor="#f8fafc", font_family="Sora")
        st.plotly_chart(fig2, use_container_width=True)

    # Distribution des charges
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 📦 Distribution des frais médicaux")
        fig3 = px.histogram(
            df_anon, x="charges", nbins=50,
            color_discrete_sequence=["#2B6CB0"],
            labels={"charges": "Frais médicaux (€)", "count": "Nombre"},
            height=380
        )
        fig3.update_layout(paper_bgcolor="white", plot_bgcolor="#f8fafc", font_family="Sora")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("#### 🌍 Frais moyens par région")
        region_avg = df_anon.groupby("region_fr")["charges"].mean().reset_index()
        region_avg.columns = ["Région", "Frais moyens (€)"]
        fig4 = px.bar(
            region_avg.sort_values("Frais moyens (€)", ascending=True),
            x="Frais moyens (€)", y="Région", orientation="h",
            color="Frais moyens (€)", color_continuous_scale="Blues",
            height=380
        )
        fig4.update_layout(paper_bgcolor="white", plot_bgcolor="#f8fafc", font_family="Sora")
        st.plotly_chart(fig4, use_container_width=True)

    # Correlation heatmap
    st.markdown("#### 🔥 Matrice de corrélation")
    df_corr = df_anon[["age", "bmi", "children", "charges"]].copy()
    corr = df_corr.corr()
    fig5 = px.imshow(
        corr,
        labels={"color": "Corrélation"},
        color_continuous_scale="RdBu",
        zmin=-1, zmax=1,
        text_auto=".2f",
        height=350
    )
    fig5.update_layout(paper_bgcolor="white", font_family="Sora")
    st.plotly_chart(fig5, use_container_width=True)


# ─── Page: Simulateur ─────────────────────────────────────────────────────────
def page_simulator(lr, le_smoker, le_sex, le_region):
    st.markdown('<div class="section-header">🔮 Simulateur de Frais Médicaux</div>', unsafe_allow_html=True)

    st.info("💡 **Transparence algorithmique** : Votre estimation est calculée par une régression linéaire "
            "interprétable. Aucune décision automatisée n'est prise — cet outil est uniquement indicatif.")

    with st.form("simulation_form", clear_on_submit=False):
        st.markdown("##### Renseignez vos informations (anonymes)")
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.slider("Âge", min_value=18, max_value=80, value=35,
                             help="Votre âge en années")
            bmi = st.number_input("IMC (kg/m²)", min_value=10.0, max_value=60.0,
                                   value=25.0, step=0.1,
                                   help="Indice de masse corporelle = poids(kg) / taille²(m)")
        with c2:
            children = st.selectbox("Nombre d'enfants", options=[0, 1, 2, 3, 4, 5], index=0)
            smoker   = st.radio("Fumeur ?", options=["Non", "Oui"], index=0,
                                 help="Avez-vous fumé au cours des 12 derniers mois ?")
        with c3:
            sex    = st.radio("Sexe", options=["Homme", "Femme"], index=0)
            region = st.selectbox("Région (US)", ["southwest", "southeast", "northwest", "northeast"])

        submitted = st.form_submit_button("🔮 Calculer mon estimation", use_container_width=True)

    if submitted:
        smoker_enc = 1 if smoker == "Oui" else 0
        sex_enc    = le_sex.transform(["male" if sex == "Homme" else "female"])[0]
        region_enc = le_region.transform([region])[0]

        X_input = np.array([[age, bmi, children, smoker_enc, sex_enc, region_enc]])
        prediction = lr.predict(X_input)[0]
        prediction = max(0, prediction)

        logger.info(f"SIMULATION | age={age} bmi={bmi} children={children} smoker={smoker} result={prediction:.2f}")

        st.markdown("")
        col_res, col_detail = st.columns([1, 1])

        with col_res:
            st.markdown(f"""
            <div class="prediction-box" role="region" aria-label="Résultat de la simulation">
                <div class="prediction-label">Estimation de vos frais médicaux annuels</div>
                <div class="prediction-amount">{prediction:,.0f} €</div>
                <div class="prediction-label" style="margin-top:1rem; font-size:0.8rem; opacity:0.8;">
                    ±15% (intervalle de confiance estimé) — À titre purement indicatif
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_detail:
            st.markdown("##### 🔍 Décomposition de l'estimation")
            coefs = {
                "Âge": lr.coef_[0] * age,
                "IMC": lr.coef_[1] * bmi,
                "Enfants": lr.coef_[2] * children,
                "Fumeur": lr.coef_[3] * smoker_enc,
                "Sexe": lr.coef_[4] * sex_enc,
                "Région": lr.coef_[5] * region_enc,
                "Constante": lr.intercept_,
            }
            contrib_df = pd.DataFrame({
                "Variable": list(coefs.keys()),
                "Contribution (€)": [round(v, 2) for v in coefs.values()]
            })
            fig = px.bar(contrib_df, x="Variable", y="Contribution (€)",
                          color="Contribution (€)",
                          color_continuous_scale="RdBu",
                          height=300)
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="#f8fafc",
                               showlegend=False, font_family="Sora")
            st.plotly_chart(fig, use_container_width=True)

        # Recommandation mutuelle
        st.markdown("##### 💊 Recommandation de contrat")
        if prediction < 5000:
            st.success("🟢 **Contrat Essentiel** — Vos frais prévisionnels sont faibles. Une couverture de base suffit.")
        elif prediction < 15000:
            st.warning("🟡 **Contrat Confort** — Couverture intermédiaire recommandée avec prise en charge partielle des spécialistes.")
        else:
            st.error("🔴 **Contrat Premium** — Vos frais prévisionnels sont élevés. Une couverture complète est fortement conseillée.")


# ─── Page: Modèle & Biais ─────────────────────────────────────────────────────
def page_model(df, lr, dt, le_smoker, le_sex, le_region, metrics, y_test, y_pred_lr, y_pred_dt):
    st.markdown('<div class="section-header">🧮 Modèle & Analyse des Biais</div>', unsafe_allow_html=True)

    # Model comparison
    tab1, tab2, tab3 = st.tabs(["📈 Performance", "⚖️ Analyse des Biais", "🌳 Interprétabilité"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Régression Linéaire")
            st.metric("R²", f"{metrics['lr']['r2']:.3f}")
            st.metric("MAE (Erreur Absolue Moyenne)", f"{metrics['lr']['mae']:,.0f} €")
            st.markdown("**Coefficients :**")
            coef_df = pd.DataFrame({
                "Variable": list(metrics['lr']['coefs'].keys()),
                "Coefficient": [round(v, 2) for v in metrics['lr']['coefs'].values()]
            })
            st.dataframe(coef_df, hide_index=True, use_container_width=True)

        with c2:
            st.markdown("##### Arbre de Décision (depth=4)")
            st.metric("R²", f"{metrics['dt']['r2']:.3f}")
            st.metric("MAE", f"{metrics['dt']['mae']:,.0f} €")
            st.info("L'arbre de décision offre une meilleure R² mais la régression linéaire "
                    "est plus facile à interpréter et expliquer aux clients.")

        # Predicted vs Actual
        st.markdown("##### Valeurs prédites vs réelles (Régression Linéaire)")
        comp_df = pd.DataFrame({"Réel": y_test.values, "Prédit": y_pred_lr})
        fig = px.scatter(comp_df, x="Réel", y="Prédit",
                          labels={"Réel": "Frais réels (€)", "Prédit": "Frais prédits (€)"},
                          opacity=0.5, color_discrete_sequence=["#2b6cb0"],
                          height=400)
        fig.add_shape(type="line", x0=0, y0=0, x1=70000, y1=70000,
                       line=dict(color="#E53E3E", dash="dash"))
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="#f8fafc", font_family="Sora")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        bias_data, df_pred = analyze_bias(df, lr, le_smoker, le_sex, le_region)
        st.markdown("##### Analyse des biais par groupe")
        st.caption("L'erreur de prédiction (réel − prédit) doit être proche de 0 pour tous les groupes.")

        for group_name, group_df in bias_data.items():
            st.markdown(f"**Groupe : {group_name}**")
            st.dataframe(group_df, use_container_width=True)

            max_err = group_df["error"].abs().max()
            if max_err > 2000:
                st.markdown(f"""<div class="bias-warning">
                    ⚠️ <strong>Biais détecté</strong> : L'erreur maximale dans le groupe <em>{group_name}</em>
                    est de <strong>{max_err:,.0f} €</strong>, ce qui dépasse le seuil acceptable.
                    <br>→ <strong>Mitigation</strong> : Rééquilibrer le dataset par sur-échantillonnage du groupe
                    sous-représenté ou pondérer les erreurs lors de l'entraînement.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="bias-ok">
                    ✅ Pas de biais significatif détecté pour <em>{group_name}</em>.
                </div>""", unsafe_allow_html=True)

        # Boxplot erreurs par fumeur
        fig_bias = px.box(df_pred, x="smoker", y="error",
                           color="smoker",
                           color_discrete_map={"yes": "#E53E3E", "no": "#2B6CB0"},
                           labels={"smoker": "Fumeur", "error": "Erreur de prédiction (€)"},
                           title="Distribution de l'erreur — Fumeurs vs Non-fumeurs",
                           height=400)
        fig_bias.update_layout(paper_bgcolor="white", plot_bgcolor="#f8fafc", font_family="Sora")
        st.plotly_chart(fig_bias, use_container_width=True)

    with tab3:
        st.markdown("##### Interprétabilité — Importance des variables")
        importance_df = pd.DataFrame({
            "Variable": ["Âge", "IMC", "Enfants", "Fumeur", "Sexe", "Région"],
            "Coefficient (€/unité)": [round(c, 2) for c in lr.coef_]
        }).sort_values("Coefficient (€/unité)", ascending=True)

        fig_imp = px.bar(
            importance_df, x="Coefficient (€/unité)", y="Variable",
            orientation="h", color="Coefficient (€/unité)",
            color_continuous_scale="RdBu",
            title="Impact de chaque variable sur les frais médicaux",
            height=400
        )
        fig_imp.update_layout(paper_bgcolor="white", plot_bgcolor="#f8fafc", font_family="Sora")
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("""
        **Lecture des coefficients :**
        - Un coefficient **positif** signifie que la variable *augmente* les frais.
        - Un coefficient **négatif** signifie qu'elle les *diminue*.
        - Exemple : si le coefficient "Fumeur" = 24 000, chaque fumeur paie en moyenne **24 000 € de plus**.
        """)


# ─── Page: RGPD & Accès ───────────────────────────────────────────────────────
def page_rgpd():
    st.markdown('<div class="section-header">📋 Conformité RGPD & Accessibilité</div>', unsafe_allow_html=True)

    with st.expander("🔒 Données collectées & finalités", expanded=True):
        st.markdown("""
| Donnée | Finalité | Durée de conservation |
|--------|----------|----------------------|
| Âge, IMC, Sexe, Statut fumeur | Estimation des frais médicaux | Durée de la session uniquement |
| Nombre d'enfants, région | Personnalisation de l'estimation | Durée de la session uniquement |
| Logs d'accès | Sécurité, audit | 12 mois maximum |


        """)


    with st.expander("📜 Droits des utilisateurs"):
        st.markdown("""
- **Droit d'accès** (Art. 15) : Vous pouvez demander quelles données vous concernent.
- **Droit de rectification** (Art. 16) : Vous pouvez corriger des informations inexactes.
- **Droit à l'effacement** (Art. 17) : Demandez la suppression de vos données.
- **Droit d'opposition** (Art. 21) : Vous pouvez vous opposer au traitement à tout moment.
- **Droit à la portabilité** (Art. 20) : Vous pouvez récupérer vos données dans un format structuré.

        """)



# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    if not check_password():
        return

    show_rgpd_consent()

    df, df_anon = load_data()
    lr, dt, le_smoker, le_sex, le_region, metrics, X_test, y_test, y_pred_lr, y_pred_dt = train_model(df)

    page = render_sidebar()

    # Header
    st.markdown("""
    <div style="background: linear-gradient(90deg,#1a365d,#2b6cb0); color:white; 
                padding:1.5rem 2rem; border-radius:16px; margin-bottom:1.5rem;
                display:flex; align-items:center; gap:1rem;">
        <div style="font-size:2.5rem;">🏥</div>
        <div>
            <h1 style="margin:0; font-size:1.8rem;">Health-InsurTech</h1>
            <p style="margin:0; opacity:0.8; font-size:0.9rem;">
                Estimation transparente & éthique de vos frais médicaux annuels
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if page == "📊 Dashboard":
        page_dashboard(df_anon, metrics)
    elif page == "🔮 Simulateur":
        page_simulator(lr, le_smoker, le_sex, le_region)
    elif page == "🧮 Modèle & Biais":
        page_model(df, lr, dt, le_smoker, le_sex, le_region, metrics, y_test, y_pred_lr, y_pred_dt)
    elif page == "📋 RGPD & Accès":
        page_rgpd()


if __name__ == "__main__":
    main()
