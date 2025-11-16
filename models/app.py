# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import time
from ga_team_selector import CricketTeamGA
from ap_opponent_analysis import run_apriori_matchups

# -------------------- PAGE CONFIG / PATHS --------------------
st.set_page_config(page_title="CIM", layout="wide", page_icon="🏆")

STATS_FILE = "D:/AI ML Cricket Project CIM model/CIM/data/player_stats_venue.csv"
ROLES_FILE = "D:/AI ML Cricket Project CIM model/CIM/data/player_roles.csv"
BALL_BY_BALL_FILE = "D:/AI ML Cricket Project CIM model/CIM/data/ball_by_ball_stats_ap.csv"

BG_PATH = "D:/AI ML Cricket Project CIM model/CIM/pictures/ipl_stadium_bg.jpg"
LOGO_PATH = "D:/AI ML Cricket Project CIM model/CIM/pictures/ipl_logo.png"

SVM_BATTER = "svm_batter_model.pkl"
SVM_BOWLER = "svm_bowler_model.pkl"
SVM_FEAT = "svm_player_performance_features.pkl"

XGB_MODEL = "xgb_delivery_model.pkl"
XGB_ENC = "xgb_delivery_label_encoders.pkl"
XGB_FEAT = "xgb_delivery_features.pkl"

ROLE_DISPLAY = {
    'opener': 'Opener',
    'middle_order': 'Middle order',
    'wicket_keeper': 'Wicket-Keeper',
    'finisher': 'Finisher',
    'spinner': 'Spinner',
    'fast_bowler': 'Fast Bowler'
}
ROLE_ORDER = ['Opener', 'Middle order', 'Wicket-Keeper', 'Finisher', 'Spinner', 'Fast Bowler']

# Team to logo mapping (filenames in LOGOS_PATH)
TEAM_LOGO_MAP = {
    'Chennai Super Kings': 'CSK.jpg',
    'Delhi Capitals': 'DC.jpg',
    'Gujarat Titans': 'GT.jpg',
    'Kolkata Knight Riders': 'KKR.jpg',
    'Lucknow Super Giants': 'LSG.jpg',
    'Mumbai Indians': 'MI.jpg',
    'Punjab Kings': 'PBKS.jpg',
    'Royal Challengers Bangalore': 'RCB.jpg',
    'Rajasthan Royals': 'RR.jpg',
    'Sunrisers Hyderabad': 'SRH.jpg',
    'csk': 'CSK.jpg',
    'dc': 'DC.jpg',
    'gt': 'GT.jpg',
    'kkr': 'KKR.jpg',
    'lsg': 'LSG.jpg',
    'mi': 'MI.jpg',
    'pbks': 'PBKS.jpg',
    'rcb': 'RCB.jpg',
    'rr': 'RR.jpg',
    'srh': 'SRH.jpg',
}
LOGOS_PATH = "D:/AI ML Cricket Project CIM model/CIM/pictures/logos"

# -------------------- SESSION STATE INIT --------------------
_default_session = {
    'best_xi': None,
    'ga_model': None,
    'input_venue': None,
    'opponent_xi': None,
    'opponent_team': None,
    'apriori_results': None,
    'svm_results': None,
    '_compute_matchups': False,
    'xgb_matchups': None,
    'input_team_display': None
}
for k, v in _default_session.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- CACHING HELPERS --------------------
@st.cache_data(show_spinner=False)
def load_csv(path):
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_svm():
    batter = joblib.load(SVM_BATTER)
    bowler = joblib.load(SVM_BOWLER)
    feats = joblib.load(SVM_FEAT)
    return batter, bowler, feats

@st.cache_resource(show_spinner=False)
def load_xgb():
    model = joblib.load(XGB_MODEL)
    enc = joblib.load(XGB_ENC)
    feat = joblib.load(XGB_FEAT)
    return model, enc, feat

# -------------------- SPLASH SCREEN --------------------
def show_splash_screen():
    st.markdown("""
        <style>
            #splash-container {
                position: fixed; top: 0; left: 0;
                width: 100vw; height: 100vh;
                background: radial-gradient(circle at center, rgba(0,0,80,1), rgba(0,0,30,1));
                display: flex; flex-direction: column;
                align-items: center; justify-content: center;
                z-index: 9999;
                animation: fadeOut 1s ease-in-out 2.5s forwards;
            }
            #splash-container h1 { color: #FFD700; font-size: 3.5rem; font-weight: 900;
                text-shadow: 3px 3px 15px rgba(255,215,0,0.8), 0 0 25px rgba(30,144,255,0.8);
                animation: glowPulse 2s infinite alternate;}
            #splash-container p { color: #F0F8FF; font-size: 1.3rem; margin-top: 10px; }
            #splash-container .icon { font-size: 3rem; animation: bounce 1.5s infinite; margin-bottom: 10px; }
            @keyframes glowPulse { from { text-shadow: 0 0 10px #FFD700; } to { text-shadow: 0 0 25px #00BFFF; } }
            @keyframes bounce { 0%,100% { transform: translateY(0);} 50% { transform: translateY(-10px);} }
            @keyframes fadeOut { from { opacity: 1; } to { opacity: 0; visibility: hidden; } }
        </style>
        <div id="splash-container">
            <div class="icon">🏏</div>
            <h1>Hybrid Cricket Intelligence Model</h1>
            <p>AI/ML-Powered IPL Analytics Platform</p>
        </div>
    """, unsafe_allow_html=True)
    time.sleep(3)

# -------------------- TEAM LOGOS / CARDS --------------------
def get_team_logo_path(team_name):
    logo_file = TEAM_LOGO_MAP.get(team_name, None) or TEAM_LOGO_MAP.get(team_name.lower(), None)
    if logo_file:
        return f"{LOGOS_PATH}/{logo_file}"
    return None

def display_team_logo(team_name, width=100):
    """Return a safe HTML string for an <img> or fallback div. Image tag is self-contained and has alt text."""
    logo_path = get_team_logo_path(team_name)
    if logo_path:
        try:
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode()
            # Use self-contained <img /> with alt, and safe styles (no stray characters)
            img_html = (
                f'<img src="data:image/jpeg;base64,{logo_b64}" alt="{team_name} logo" '
                f'width="{width}" style="margin:5px; border-radius:20px; box-shadow:0 0 30px rgba(255,215,0,0.8),'
                f'0 0 60px rgba(30,144,255,0.6); filter:brightness(1.1) contrast(1.15) saturate(1.2);'
                f'transition:all 0.3s ease-in-out; cursor:pointer; border:2px solid rgba(255,215,0,0.8);'
                f'backdrop-filter: blur(10px);" />'
            )
            return img_html
        except Exception:
            pass
    # fallback square badge with initials
    initials = (team_name[:3].upper() if team_name else "TBD")
    fallback = (
        f'<div style="width:{width}px; height:{width}px; background:linear-gradient(135deg, #FFD700, #FFA500); '
        f'border-radius:20px; display:flex; align-items:center; justify-content:center; color:#000; font-weight:bold; '
        f'font-size:24px; box-shadow:0 0 30px rgba(255,215,0,0.8); border:2px solid rgba(255,215,0,0.9);">'
        f'{initials}</div>'
    )
    return fallback

def display_team_card(team_name, logo_width=180):
    """Directly renders a luxury-styled team-card (logo inside) to Streamlit."""
    logo_html = display_team_logo(team_name, width=logo_width)
    # Balanced CSS and HTML; no stray closing tags outside this block
    st.markdown(f"""
        <style>
            @keyframes luxGlow {{ 0% {{ box-shadow: 0 0 20px rgba(255,215,0,0.5); }} 50% {{ box-shadow: 0 0 40px rgba(255,215,0,0.8); }} 100% {{ box-shadow: 0 0 20px rgba(255,215,0,0.5); }} }}
            .luxury-team-card {{
                background: linear-gradient(135deg, rgba(30,144,255,0.15), rgba(25,25,112,0.4));
                border:2px solid rgba(255,215,0,0.8);
                border-radius:20px; padding:20px; text-align:center;
                box-shadow:0 0 30px rgba(255,215,0,0.6), 0 0 60px rgba(30,144,255,0.4);
                animation:luxGlow 3s ease-in-out infinite; backdrop-filter: blur(12px);
                display:flex; justify-content:center; align-items:center; min-height:160px;
            }}
            .luxury-logo-container {{ z-index:1; }}
        </style>
        <div class="luxury-team-card">
            <div class="luxury-logo-container">{logo_html}</div>
        </div>
    """, unsafe_allow_html=True)


# -------------------- BACKGROUND + HEADER + THEME --------------------
def add_bg_logo(bg_path=BG_PATH, logo_path=LOGO_PATH):
    try:
        with open(bg_path, "rb") as f:
            bg_b64 = base64.b64encode(f.read()).decode()
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
            <style>
            .stApp {{
              background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url("data:image/jpg;base64,{bg_b64}");
              background-size: cover; background-position: center; background-attachment: fixed;
            }}
            .app-header {{ text-align:center; margin-top:10px; margin-bottom:-10px; }}
            .app-header img {{ border-radius: 8px; }}
            </style>
            <div class="app-header"><img src="data:image/png;base64,{logo_b64}" width="120" alt="CIM logo" /></div>
        """, unsafe_allow_html=True)
    except Exception:
        st.warning("Background / logo not found (check BG_PATH / LOGO_PATH).")


def inject_css():
    st.markdown("""
    <style>
    * { box-sizing: border-box; }
    html, body, [class*="css"]  { font-family: 'Inter', 'Segoe UI', 'Roboto', sans-serif; color: #FFFFFF; background: linear-gradient(135deg,#0a0e27 0%, #1a1f3a 100%); }

    h1, h2, h3 { color: #FFD700; text-shadow: 2px 2px 8px rgba(0,0,0,0.8); font-weight:800; letter-spacing:0.5px; }
    h2, h3 { color: #FFFFFF !important; }

    .metric-card { background: linear-gradient(135deg, rgba(25,25,112,0.5), rgba(30,60,120,0.3)); border:2px solid rgba(255,215,0,0.6); border-radius:15px; padding:25px; margin-top:15px; color:#F0F8FF; box-shadow:0 8px 32px rgba(0,0,0,0.3); backdrop-filter: blur(10px); position:relative; overflow:hidden; }

    /* Alerts */
    div[data-testid*="stAlert"], div[data-testid*="stAlert"] * { color: #FFFFFF !important; text-shadow:1px 1px 3px rgba(0,0,0,0.9); font-weight:600; }
    div[data-testid*="stSuccess"] { background: linear-gradient(90deg,#00C851, #007E33) !important; border:2px solid #00FF7F !important; color: #FFFFFF !important; }
    div[data-testid*="stInfo"] { background: linear-gradient(90deg,#0099FF, #0056D2) !important; border:2px solid #00BFFF !important; color:#FFFFFF !important; }
    div[data-testid*="stWarning"] { background: linear-gradient(90deg,#FFD700, #FFB800) !important; color:#000000 !important; }
    div[data-testid*="stError"] { background: linear-gradient(90deg,#FF4444, #CC0000) !important; color:#FFFFFF !important; }

    /* Tabs */
    div[data-baseweb="tab-list"] { display:flex !important; justify-content:space-evenly !important; align-items:center !important; width:100% !important; background: linear-gradient(90deg, rgba(0,0,64,0.3), rgba(0,50,100,0.3)); border-radius:12px; border:2px solid rgba(255,215,0,0.5); padding:8px 0; margin-bottom:20px; backdrop-filter: blur(10px); }
    button[data-baseweb="tab"] { flex-grow:1 !important; justify-content:center !important; font-size:15px !important; font-weight:700 !important; padding:12px 8px !important; background:transparent !important; border:none !important; border-radius:8px !important; color: #FFFFFF !important; transition: all 0.3s; margin:0 2px; }
    button[data-baseweb="tab"]:hover { background: rgba(255,215,0,0.1) !important; color: #FFD700 !important; transform: translateY(-2px); }
    button[data-baseweb="tab"][aria-selected="true"] { background: linear-gradient(135deg, rgba(30,144,255,0.4), rgba(0,100,200,0.3)) !important; border-bottom:3px solid #FFD700 !important; color:#FFFFFF !important; }

    /* Buttons */
    div.stButton > button { background: linear-gradient(135deg,#1E90FF 0%, #00BFFF 50%); color:white; border-radius:10px; height:2.8em; width:100%; font-weight:bold; font-size:15px; border:2px solid rgba(255,215,0,0.5); box-shadow: 0 4px 15px rgba(30,144,255,0.5); transition: all 0.3s; }
    div.stButton > button:hover { background: linear-gradient(135deg,#FFD700 0%, #FFA500 50%); color:black; transform: translateY(-2px); box-shadow: 0 8px 25px rgba(255,215,0,0.8); }

    /* Inputs */
    div[data-baseweb="select"], div[data-testid*="multiselect"] { background: rgba(30,144,255,0.08) !important; border:2px solid rgba(255,215,0,0.4) !important; border-radius:8px !important; }
    input, textarea, div[data-baseweb="base-input"] { background: rgba(30,144,255,0.06) !important; border:2px solid rgba(255,215,0,0.4) !important; color:white !important; border-radius:8px !important; }
    input::placeholder, textarea::placeholder { color: rgba(255,255,255,0.6) !important; }
    
    /* Selectbox and Multiselect Labels - Royal Blue */
    label, .stSelectbox label, .stMultiselect label { color: #4169E1 !important; font-weight: 700 !important; }
    div[data-testid="stSelectbox"] label, div[data-testid="stMultiSelect"] label { color: #4169E1 !important; font-weight: 700 !important; }

    /* Tables */
    th { background: linear-gradient(135deg,#1E90FF 0%, #00BFFF 50%); color:#FFFFFF !important; font-weight:bold !important; }
    td { background-color: rgba(255,255,255,0.88) !important; color:#1a1a1a !important; }

    div[data-testid="stDataFrame"] { border:2px solid rgba(255,215,0,0.3) !important; border-radius:10px !important; overflow:hidden !important; box-shadow:0 4px 15px rgba(0,0,0,0.2) !important; }

    footer { visibility: hidden; }

    /* Make any markdown text strongly visible */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span { color: #FFFFFF !important; opacity: 1.0 !important; text-shadow:1px 1px 3px rgba(0,0,0,0.9); font-weight:600; }

    /* Footer adjustments (white text) */
    .dashboard-footer { color: #FFFFFF !important; }
    .dashboard-footer p { color: #FFFFFF !important; }
    </style>
    """, unsafe_allow_html=True)


# Show splash once per session
if "splash_done" not in st.session_state:
    show_splash_screen()
    st.session_state.splash_done = True

# Header / background
add_bg_logo()
inject_css()

st.markdown("""
    <div style="text-align:center; padding: 8px 0 20px;">
        <h1 style="margin:0; color:#FFD700;">🏏 Hybrid Cricket Intelligence Model</h1>
        <p style="margin:6px 0 0; color:#E0E0E0;">AI/ML-Powered Decision Support System for IPL Team Strategy</p>
        <hr style="border:2px solid #FFD700; width:70%; margin:12px auto;">
    </div>
""", unsafe_allow_html=True)

# -------------------- UTILS --------------------
def format_roles(df):
    df = df.copy()
    if 'role' in df.columns:
        df['role'] = df['role'].map(ROLE_DISPLAY).fillna(df['role'])
    return df

def sort_by_role(df):
    df = df.copy()
    if 'role' in df.columns:
        df['role_order_index'] = df['role'].map(lambda r: ROLE_ORDER.index(r) if r in ROLE_ORDER else 99)
        df = df.sort_values(by='role_order_index').drop(columns=['role_order_index'])
    return df

def format_floats(df):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].apply(lambda x: int(x) if pd.notnull(x) and float(x).is_integer() else round(x,2) if pd.notnull(x) else x)
    return df

def clean_table(df):
    cols_to_drop = [c for c in df.columns if 'lower' in c or 'venue' in c or c == 'franchise']
    return df.drop(columns=cols_to_drop, errors='ignore')

def show_table(df, start_index=1):
    df = df.copy()
    df.index = range(start_index, start_index + len(df))
    st.dataframe(df, use_container_width=True)

# -------------------- MAIN TABS --------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ GA Team Selection",
    "📊 SVM Player Performance",
    "🧩 Apriori Opponent Analysis",
    "🎯 XGBoost Matchup Prediction"
])

# Load core CSVs (cached)
try:
    roles_df = load_csv(ROLES_FILE)
    stats_df = load_csv(STATS_FILE)
except Exception as e:
    st.error(f"Could not load core CSV files: {e}")
    st.stop()

franchise_display = sorted(roles_df['franchise'].dropna().unique())
venue_display = sorted(stats_df['venue'].dropna().unique())

# =======================================================
# TAB 1: GA TEAM SELECTION
# =======================================================
with tab1:
    st.subheader("⚙️ Best XI Selector (Genetic Algorithm)")
    col1, col2 = st.columns([2, 1])
    with col1:
        input_team_display = st.selectbox("Select Franchise", franchise_display, key="ga_franchise")
        input_venue_display = st.selectbox("Select Venue", venue_display, key="ga_venue")
    with col2:
        st.markdown("### 🏏 Team Badge")
        display_team_card(input_team_display, logo_width=100)

    st.session_state['input_venue'] = input_venue_display
    st.session_state['input_team_display'] = input_team_display

    input_team = input_team_display.strip().lower()
    input_venue = input_venue_display.strip().lower()

    if st.button("🏆 Generate Best XI", key="ga_generate"):
        try:
            ga_model = CricketTeamGA(STATS_FILE, ROLES_FILE)
            best_team = ga_model.run_ga(input_team, input_venue)

            st.session_state['best_xi'] = best_team
            st.session_state['ga_model'] = ga_model

            st.subheader(f"{input_team_display} Squad List")
            squad_df = clean_table(ga_model.franchise_list.copy())
            squad_df = format_floats(sort_by_role(format_roles(squad_df)))
            show_table(squad_df, start_index=1)

            st.subheader(f"Player Pool Used (min_matches ≥ {ga_model.min_matches_used})")
            pool_df = clean_table(ga_model.player_pool.copy())
            pool_df = format_floats(sort_by_role(format_roles(pool_df)))
            show_table(pool_df, start_index=1)

            st.subheader("Role Counts")
            role_counts_data = [
                {"Role": ROLE_DISPLAY[r], "Count": int(ga_model.role_counts.get(r, 0))}
                for r in ['opener','middle_order','wicket_keeper','finisher','spinner','fast_bowler']
            ]
            role_counts_df = pd.DataFrame(role_counts_data)
            show_table(role_counts_df, start_index=1)

            best_team_df = clean_table(best_team.copy())
            best_team_df = format_floats(sort_by_role(format_roles(best_team_df)))
            st.subheader(f'Selected Best Playing XI for {input_venue_display}')

            col_logo, col_xi = st.columns([1, 4])
            with col_logo:
                display_team_card(input_team_display, logo_width=120)
            with col_xi:
                show_table(best_team_df, start_index=1)

            leftover_df = ga_model.player_pool[~ga_model.player_pool['player_name'].isin(best_team['player_name'])].copy()
            if not leftover_df.empty:
                leftover_df = leftover_df.groupby('player_name', as_index=False).agg({
                    'role': 'first', 'matches': 'sum', 'runs': 'sum', 'bat_avg': 'mean', 'bat_sr': 'mean',
                    'wickets': 'sum', 'econ': 'mean', 'indian': 'first'
                })
            leftover_df = clean_table(leftover_df)
            leftover_df = format_floats(sort_by_role(format_roles(leftover_df)))
            st.subheader("Players Left Out from Player Pool")
            show_table(leftover_df, start_index=len(best_team_df)+1)

            st.success(f"Total Fitness Score: {ga_model.fitness(best_team):.2f}")
            if "fallback" in str(ga_model.min_matches_used).lower():
                st.info("Not enough venue stats, showing best squad XI (aggregated stats) instead.")

        except Exception as e:
            st.error(f"GA error: {e}")

    if st.session_state['best_xi'] is not None:
        st.info("✅ Best XI available in session (Tabs 2-4 can use it).")

# =======================================================
# TAB 2: SVM PLAYER PERFORMANCE
# =======================================================
with tab2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("📊 Playing XI Performance Prediction (SVM Model)")

    if st.session_state['best_xi'] is None:
        st.warning("Run the GA Team Selection (Tab 1) first.")
    else:
        if st.button("🔮 Generate Playing XI Performance Prediction", key="svm_generate"):
            try:
                batter_model, bowler_model, feature_dict = load_svm()
            except Exception as e:
                st.error(f"Could not load SVM models: {e}")
                st.stop()

            best_xi_df = st.session_state['best_xi'].copy()
            best_xi_df['venue'] = st.session_state['input_venue'] or ''
            best_xi_df['role'] = best_xi_df['role'].str.lower()

            def classify_group(r):
                if any(x in r for x in ['bowler', 'spinner']): return 'bowler'
                return 'batter'

            best_xi_df['group'] = best_xi_df['role'].apply(classify_group)
            preds_all = []

            batters = best_xi_df[best_xi_df['group']=='batter'].copy()
            if not batters.empty:
                Xb = batters[['runs','bat_avg','bat_sr']].copy()
                Xb = pd.concat([Xb, pd.get_dummies(batters[['venue']], drop_first=True)], axis=1)
                for col in feature_dict['batter']:
                    if col not in Xb.columns: Xb[col] = 0
                Xb = Xb[feature_dict['batter']]
                batters['Predicted_Performance'] = batter_model.predict(Xb)
                preds_all.append(batters)

            bowlers = best_xi_df[best_xi_df['group']=='bowler'].copy()
            if not bowlers.empty:
                Xw = bowlers[['wickets','econ','runs']].copy()
                Xw = pd.concat([Xw, pd.get_dummies(bowlers[['venue']], drop_first=True)], axis=1)
                for col in feature_dict['bowler']:
                    if col not in Xw.columns: Xw[col] = 0
                Xw = Xw[feature_dict['bowler']]
                bowlers['Predicted_Performance'] = bowler_model.predict(Xw)
                preds_all.append(bowlers)

            if preds_all:
                final_df = pd.concat(preds_all).sort_index()
                show_df = final_df[['player_name','role','matches','runs','bat_avg','bat_sr','wickets','econ','Predicted_Performance']]
                show_df = format_floats(sort_by_role(format_roles(show_df)))
                st.dataframe(show_df, use_container_width=True)
                st.session_state['svm_results'] = show_df
                summary = show_df['Predicted_Performance'].value_counts().reset_index()
                summary.columns = ['Performance Category','Count']
                st.subheader("Performance Summary")
                st.dataframe(summary, use_container_width=True)
                st.success("✅ Playing XI predictions generated successfully!")
            else:
                st.warning("No players to predict.")
        elif st.session_state['svm_results'] is not None:
            st.dataframe(st.session_state['svm_results'])
    st.markdown("</div>", unsafe_allow_html=True)

# =======================================================
# TAB 3: APRIORI OPPONENT ANALYSIS
# =======================================================
with tab3:
    st.subheader("🧩 Opponent Analysis - Matchup Insights (Apriori Algorithm)")
    st.info("Select opponent team and XI to find historical matchup patterns.")

    if st.session_state['best_xi'] is None:
        st.warning("Run the GA Team Selection (Tab 1) first.")
    else:
        df_roles_raw = roles_df
        franchise_display = sorted(df_roles_raw['franchise'].dropna().unique())

        col1, col2 = st.columns([3, 1])
        with col1:
            opponent_team = st.selectbox("Select Opponent Franchise", franchise_display, key="apriori_opponent_team")
            # persist opponent team selection to session so other tabs (XGBoost) can use it
            st.session_state['opponent_team'] = opponent_team
        with col2:
            st.markdown("### 🏏 Opponent Badge")
            display_team_card(opponent_team, logo_width=100)

        opponent_squad = df_roles_raw[df_roles_raw['franchise']==opponent_team]['player_name'].tolist()
        opponent_xi = st.multiselect("Select Opponent Playing XI (11 players)", opponent_squad, key="apriori_opponent_xi")

        if st.button("Run Apriori Matchup Analysis", key="apriori_run"):
            if not opponent_xi or len(opponent_xi) != 11:
                st.warning("Please select exactly 11 players for opponent XI.")
            else:
                try:
                    ball_df = load_csv(BALL_BY_BALL_FILE)
                    results_df = run_apriori_matchups(my_xi=st.session_state['best_xi'], opponent_xi=opponent_xi, ball_df=ball_df)
                    st.session_state['apriori_results'] = results_df
                    st.session_state['opponent_xi'] = opponent_xi
                    st.success("✅ Apriori analysis complete - results stored in session.")
                except Exception as e:
                    st.error(f"Apriori error: {e}")

        if st.session_state['apriori_results'] is not None:
            results_df = st.session_state['apriori_results']
            if results_df.empty:
                st.warning("No strong historical patterns found.")
            else:
                ball_df = load_csv(BALL_BY_BALL_FILE)

                def extract_parts_with_case(row):
                    ant = [s.strip() for s in row['antecedents'].split(',')]
                    cons = [s.strip() for s in row['consequents'].split(',')]
                    bowler = next((x.replace('bowler:', '').strip() for x in ant if x.startswith('bowler:')), '')
                    batsman = next((x.replace('batsman:', '').strip() for x in cons if x.startswith('batsman:')), '')
                    venue = next((x.replace('venue:', '').strip() for x in ant if x.startswith('venue:')), '')
                    phase = next((x.replace('phase:', '').strip() for x in ant if x.startswith('phase:')), '')
                    dismissal_type = next((x.replace('dismissal:', '').strip() for x in ant if x.startswith('dismissal:')), '')

                    bowler_case = ball_df.loc[ball_df['bowler'].str.lower() == bowler.lower(), 'bowler'].head(1)
                    batsman_case = ball_df.loc[ball_df['batsman'].str.lower() == batsman.lower(), 'batsman'].head(1)
                    venue_case = ball_df.loc[ball_df['venue'].str.lower() == venue.lower(), 'venue'].head(1)

                    bowler = bowler_case.iloc[0] if not bowler_case.empty else bowler.title()
                    batsman = batsman_case.iloc[0] if not batsman_case.empty else batsman.title()
                    venue = venue_case.iloc[0] if not venue_case.empty else venue.title()
                    phase = phase.capitalize() if phase else ''

                    return pd.Series({
                        'Bowler': bowler,
                        'Batsman': batsman,
                        'Venue': venue,
                        'Phase': phase,
                        'Dismissal': dismissal_type,
                        'Support': row.get('support', None),
                        'Confidence': row.get('confidence', None),
                        'Lift': row.get('lift', None)
                    })

                pretty_df = results_df.apply(extract_parts_with_case, axis=1)
                deduped = pretty_df.sort_values(['Bowler','Batsman','Lift','Confidence'], ascending=[True,True,False,False]).drop_duplicates(['Bowler','Batsman'])
                top_results = deduped.sort_values(['Lift','Confidence','Support'], ascending=False).head(20).reset_index(drop=True)
                top_results['Sl.No'] = range(1, len(top_results)+1)
                cols = ['Sl.No'] + [c for c in top_results.columns if c != 'Sl.No']
                top_results = top_results[cols]
                st.subheader("🔝 Best Unique Bowler-Batsman Matchups")
                st.dataframe(top_results.set_index('Sl.No'))

                st.subheader("📝 Tactical Insights")
                for i, row in top_results.iterrows():
                    if row.Phase and row.Venue:
                        ctx = f"in the {row.Phase} overs at {row.Venue}"
                    elif row.Phase:
                        ctx = f"in the {row.Phase} overs"
                    elif row.Venue:
                        ctx = f"at {row.Venue}"
                    else:
                        ctx = ""
                    sentence = f"{row.Bowler} bowling {ctx} has a high historical success against {row.Batsman}." if ctx else f"{row.Bowler} bowling has a high historical success against {row.Batsman}."
                    st.write(f"{i+1}. {sentence}")

# =======================================================
# TAB 4: XGBOOST MATCHUP PREDICTION
# =======================================================
with tab4:
    st.subheader("🎯 Matchup Dismissal Prediction (XGBoost Model)")
    st.info("View bowler-batsman dismissal probabilities based on historical IPL data.")

    if not (st.session_state['best_xi'] is not None and st.session_state['opponent_xi'] is not None and st.session_state['input_venue'] is not None):
        st.warning("Run GA (Tab 1) and Apriori (Tab 3) first: Best XI, Opponent XI and Venue are required.")
    else:
        try:
            xgb_model, xgb_encoders, xgb_feat_cols = load_xgb()
        except Exception as e:
            st.error(f"Could not load XGBoost artifacts: {e}")
            st.stop()

        stats_df_local = load_csv(BALL_BY_BALL_FILE)
        if "is_wicket" not in stats_df_local.columns:
            stats_df_local["is_wicket"] = (~stats_df_local.get("dismissal_type", pd.Series([np.nan]*len(stats_df_local))).isna()).astype(int)

        try:
            roles_df_local = load_csv(ROLES_FILE)
            roles_map = dict(zip(roles_df_local['player_name'].str.lower(), roles_df_local['role'].str.lower()))
        except Exception:
            roles_map = {}

        ROLE_ORDER_SHORT = ['opener','middle_order','wicket_keeper','finisher','spinner','fast_bowler']
        ROLE_LABEL = {'opener':'Opener','middle_order':'Middle','wicket_keeper':'WK','finisher':'Finisher','spinner':'Spinner','fast_bowler':'Fast'}

        def order_xi(df_or_list):
            players=[]
            if isinstance(df_or_list, pd.DataFrame):
                df = df_or_list.copy()
                if 'role' in df.columns:
                    df['role'] = df['role'].fillna('').astype(str).str.lower()
                df['player_name'] = df['player_name'].astype(str).str.strip()
                for r in ROLE_ORDER_SHORT:
                    selected = df[df['role']==r]['player_name'].tolist()
                    players.extend(selected)
                rest = [p for p in df['player_name'].tolist() if p not in players]
                players.extend(rest)
            else:
                names = [str(x).strip() for x in df_or_list]
                grouped = {r:[] for r in ROLE_ORDER_SHORT}
                unknown = []
                for n in names:
                    r = roles_map.get(n.lower(), '')
                    if r in ROLE_ORDER_SHORT:
                        grouped[r].append(n)
                    else:
                        unknown.append(n)
                for r in ROLE_ORDER_SHORT:
                    players.extend(grouped[r])
                players.extend(unknown)
            seen=set(); out=[]
            for p in players:
                if p not in seen:
                    out.append(p); seen.add(p)
            return out

        best_xi_df = st.session_state['best_xi'].copy()
        opponent_list = st.session_state['opponent_xi'].copy()
        venue = st.session_state['input_venue']

        my_xi_ordered = order_xi(best_xi_df)
        opp_xi_ordered = order_xi(opponent_list)

        # ===== COLLAPSIBLE SECTION 1: TEAM INFO =====
        with st.expander("👥 **Teams & Venue Information**", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("##### 🏏 Your Team")
                your_team_name = st.session_state.get('input_team_display', 'Your Team')
                display_team_card(your_team_name, logo_width=80)
                with st.expander("View XI", expanded=False):
                    for p in my_xi_ordered:
                        r = (best_xi_df.loc[best_xi_df['player_name']==p, 'role'].iloc[0] if p in best_xi_df['player_name'].tolist() and 'role' in best_xi_df.columns else roles_map.get(p.lower(), ''))
                        label = ROLE_LABEL.get(r, r.title() if r else '')
                        st.write(f"• **{p}** {f'({label})' if label else ''}")
            
            with col2:
                st.markdown("##### 🎯 Opponent Team")
                display_team_card(st.session_state.get('opponent_team', 'Opponent'), logo_width=80)
                with st.expander("View XI", expanded=False):
                    for p in opp_xi_ordered:
                        r = roles_map.get(p.lower(), '')
                        label = ROLE_LABEL.get(r, r.title() if r else '')
                        st.write(f"• {p} {f'({label})' if label else ''}")
            
            with col3:
                st.markdown("##### 📍 Match Details")
                st.info(f"**Venue:** {venue}")
                phase_list = sorted(stats_df_local['phase'].dropna().unique())
                phase = st.selectbox("**Phase:**", phase_list, index=0, key="xgb_phase")

        # ===== COLLAPSIBLE SECTION 2: MATCHUP ANALYSIS =====
        with st.expander("⚙️ **Compute & Configure Analysis**", expanded=True):
            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                if st.button("🚀 Compute Matchups", key="xgb_compute", use_container_width=True):
                    st.session_state['_compute_matchups'] = True
                    st.success("✅ Processing matchups...")
            
            with col_right:
                st.markdown("Select which XI you want to analyze against opposing bowlers.")
        
        # ===== COLLAPSIBLE SECTION 3: RESULTS =====
        with st.expander("📊 **Player Matchup Analysis**", expanded=True):
            if st.session_state.get('_compute_matchups', False):
                col_select, col_filter = st.columns([2, 1])
                
                with col_select:
                    inspect_side = st.radio("📋 Choose XI to inspect", ("Your XI", "Opponent XI"), horizontal=True, key="xgb_inspect_side")
                    if inspect_side == "Your XI":
                        sel_player = st.selectbox("Select batsman from your XI", my_xi_ordered, key="xgb_sel_player")
                        inspect_vs = opp_xi_ordered
                    else:
                        sel_player = st.selectbox("Select batsman from opponent XI", opp_xi_ordered, key="xgb_sel_player_opp")
                        inspect_vs = my_xi_ordered
                
                with col_filter:
                    top_n = st.slider("Top bowlers to show", min_value=3, max_value=15, value=6, key="xgb_top_n")
                
                # Compute matchups
                @st.cache_data(show_spinner=False)
                def compute_probs_for_pair(batsman, bowlers_list, venue, phase, stats_df_arg, _model, _encoders, feat_cols):
                    rows=[]
                    for bowler in bowlers_list:
                        b = str(batsman).lower(); w = str(bowler).lower(); v = str(venue).lower(); p = str(phase).lower()
                        sub_bats = stats_df_arg[stats_df_arg['batsman'].str.lower()==b] if 'batsman' in stats_df_arg.columns else pd.DataFrame()
                        recent_form = 0.0
                        if 'match_id' in sub_bats.columns and not sub_bats.empty:
                            try:
                                match_runs = sub_bats.groupby('match_id')['runs_scored'].sum().reset_index().sort_values('match_id')
                                recent_form = match_runs['runs_scored'].shift(1).rolling(3, min_periods=1).mean().iloc[-1]
                                if np.isnan(recent_form): recent_form = 0.0
                            except Exception:
                                recent_form = 0.0
                        bw = stats_df_arg[stats_df_arg['bowler'].str.lower()==w] if 'bowler' in stats_df_arg.columns else pd.DataFrame()
                        bowler_wickets_last50 = float(bw['is_wicket'].shift(1).tail(50).sum()) if not bw.empty else 0.0
                        hv = stats_df_arg[(stats_df_arg['batsman'].str.lower()==b)&(stats_df_arg['bowler'].str.lower()==w)] if 'batsman' in stats_df_arg.columns else pd.DataFrame()
                        batsman_runs_vs_bowler_last50 = float(hv['runs_scored'].shift(1).tail(50).sum()) if not hv.empty else 0.0
                        vb = stats_df_arg[(stats_df_arg['batsman'].str.lower()==b)&(stats_df_arg['phase'].str.lower()==p)] if 'phase' in stats_df_arg.columns else pd.DataFrame()
                        bat_phase_rpb = float(vb['runs_scored'].sum() / (len(vb) if len(vb)>0 else 1))
                        bwp = stats_df_arg[(stats_df_arg['bowler'].str.lower()==w)&(stats_df_arg['phase'].str.lower()==p)] if 'phase' in stats_df_arg.columns else pd.DataFrame()
                        bp_wicket_rate = float(bwp['is_wicket'].sum() / (len(bwp) if len(bwp)>0 else 1))
                        row = {
                            'batsman_l': b, 'bowler_l': w, 'venue_l': v, 'phase_l': p,
                            'recent_bat_form': float(recent_form),
                            'bowler_wickets_last50': float(bowler_wickets_last50),
                            'batsman_runs_vs_bowler_last50': float(batsman_runs_vs_bowler_last50),
                            'bat_phase_rpb': float(bat_phase_rpb),
                            'bp_wicket_rate': float(bp_wicket_rate)
                        }
                        for f in feat_cols:
                            if f not in row: row[f] = 0.0
                        df_row = pd.DataFrame([row])[feat_cols]
                        for col, le in _encoders.items():
                            if col in df_row.columns:
                                val = str(df_row[col].iloc[0])
                                try:
                                    df_row[col] = le.transform([val]) if val in le.classes_ else 0
                                except Exception:
                                    try:
                                        df_row[col] = le.transform([val])
                                    except Exception:
                                        df_row[col] = 0
                        df_row = df_row.astype(float).fillna(0)
                        try:
                            prob = float(_model.predict_proba(df_row)[0][1]) * 100.0
                        except Exception:
                            prob = np.nan
                        rows.append((bowler, prob))
                    rows = sorted(rows, key=lambda x: (0 if np.isnan(x[1]) else x[1]), reverse=True)
                    return rows

                with st.spinner("⏳ Calculating probabilities..."):
                    pair_probs = compute_probs_for_pair(sel_player, inspect_vs, venue, phase, stats_df_local, _model=xgb_model, _encoders=xgb_encoders, feat_cols=xgb_feat_cols)
                    df_top = pd.DataFrame(pair_probs, columns=["Bowler","Dismissal %"])
                    df_top["Dismissal %"] = df_top["Dismissal %"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "–")
                    st.session_state['xgb_matchups'] = df_top

                # Display results
                st.markdown(f"### 🎯 Top {top_n} Dismissal Threats vs **{sel_player}**")
                st.markdown(f"*At {venue} during {phase} phase*")
                
                result_col1, result_col2 = st.columns([2, 1])
                with result_col1:
                    st.table(st.session_state['xgb_matchups'].head(top_n))
                
                with result_col2:
                    st.markdown("#### 📈 Summary")
                    st.metric("Total Bowlers Analyzed", len(st.session_state['xgb_matchups']))
                    if not st.session_state['xgb_matchups'].empty:
                        max_threat = st.session_state['xgb_matchups'].iloc[0]
                        st.metric("Highest Threat", max_threat['Bowler'], f"{max_threat['Dismissal %']}% risk")

                with st.expander("📋 Show All Matchups"):
                    st.dataframe(st.session_state['xgb_matchups'], use_container_width=True)
            else:
                st.info("👆 Click **Compute Matchups** above to generate predictions!")

# -------------------- FOOTER --------------------
st.markdown("""
<style>
    .dashboard-footer { background: linear-gradient(90deg, rgba(30,144,255,0.05), rgba(255,215,0,0.02)); border-top:2px solid rgba(255,215,0,0.4); padding:20px; margin-top:40px; text-align:center; color:#FFFFFF; border-radius:10px; font-size:13px; backdrop-filter: blur(10px); }
    .dashboard-footer p { color: #FFFFFF; margin: 6px 0; }
</style>
<div class="dashboard-footer">
    <p>© 2025 Hybrid Cricket Intelligence Model (CIM) | Developed by Santhoji V</p>
    <p style="font-size:11px; margin-top:8px;">AI/ML-Powered Decision Support System for IPL Team Strategy</p>
</div>
""", unsafe_allow_html=True)
