#app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from ga_team_selector import CricketTeamGA
from ap_opponent_analysis import run_apriori_matchups

import streamlit as st
import time

# -------------------- IPL Splash Screen --------------------
def show_splash_screen():
    st.markdown("""
        <style>
            /* Fullscreen splash container */
            #splash-container {
                position: fixed;
                top: 0; left: 0;
                width: 100vw; height: 100vh;
                background: radial-gradient(circle at center, rgba(0,0,80,1), rgba(0,0,30,1));
                display: flex; flex-direction: column;
                align-items: center; justify-content: center;
                z-index: 9999;
                animation: fadeOut 1s ease-in-out 2.5s forwards;
            }

            /* IPL Title Styling */
            #splash-container h1 {
                color: #FFD700;
                font-size: 3.5rem;
                font-weight: 900;
                text-shadow: 3px 3px 15px rgba(255,215,0,0.8),
                             0 0 25px rgba(30,144,255,0.8);
                animation: glowPulse 2s infinite alternate;
            }

            /* Subtitle */
            #splash-container p {
                color: #F0F8FF;
                font-size: 1.3rem;
                margin-top: 10px;
                letter-spacing: 0.5px;
            }

            /* Cricket stumps icon / emoji bounce */
            #splash-container .icon {
                font-size: 3rem;
                animation: bounce 1.5s infinite;
                margin-bottom: 10px;
            }

            @keyframes glowPulse {
                from { text-shadow: 0 0 10px #FFD700; }
                to { text-shadow: 0 0 25px #00BFFF; }
            }

            @keyframes bounce {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-10px); }
            }

            @keyframes fadeOut {
                from { opacity: 1; }
                to { opacity: 0; visibility: hidden; }
            }
        </style>

        <div id="splash-container">
            <div class="icon">🏏</div>
            <h1>Hybrid Cricket Intelligence Model</h1>
            <p>AI/ML-Powered IPL Analytics Platform</p>
        </div>
    """, unsafe_allow_html=True)

    # Wait for splash duration before continuing
    time.sleep(3)


# -------------------- CONFIG / PATHS --------------------
st.set_page_config(page_title="CIM", layout="wide", page_icon="D:/AI ML Cricket Project CIM model/CIM/pictures/ipl_logo.png")

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

# -------------------- SESSION STATE INIT --------------------
_default_session = {
    'best_xi': None,
    'ga_model': None,
    'input_venue': None,
    'opponent_xi': None,
    'apriori_results': None,
    'svm_results': None,
    '_compute_matchups': False,
    'xgb_matchups': None
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

# -------------------- UI UTILITIES --------------------
def add_bg_logo(bg_path=BG_PATH, logo_path=LOGO_PATH):
    try:
        with open(bg_path, "rb") as f:
            bg_b64 = base64.b64encode(f.read()).decode()
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        st.markdown(f"""
            <style>
            .stApp {{
              background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                              url("data:image/jpg;base64,{bg_b64}");
              background-size: cover;
              background-position: center;
              background-attachment: fixed;
            }}
            .app-header {{ text-align:center; margin-top:10px; margin-bottom:-10px; }}
            </style>
            <div class="app-header"><img src="data:image/png;base64,{logo_b64}" width="130"></div>
        """, unsafe_allow_html=True)
    except Exception:
        st.warning("Background / logo not found (check BG_PATH / LOGO_PATH).")

# -------------------- GLOBAL IPL STYLE THEME --------------------
def inject_css():
    st.markdown("""
    <style>
    /* Overall font and text enhancements */
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        color: #FFFFFF;
    }

    /* Headings with glowing gold effect */
    h1, h2, h3 {
        color: #FFD700;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
        font-weight: 800;
    }

    /* Cards with transparent royal blue & golden border */
    .metric-card {
        background: rgba(25, 25, 112, 0.6);
        border: 2px solid rgba(255, 215, 0, 0.7);
        border-radius: 18px;
        padding: 20px;
        margin-top: 15px;
        color: #F0F8FF;
        box-shadow: 0 0 15px rgba(255,215,0,0.4);
    }

    /* ===== 🧾 IPL-Themed Alert Boxes with Better Visibility ===== */

    /* Target all possible alert containers (outer + inner layers) */
    div[data-testid="stAlert"], 
    div[data-testid="stAlert"] * ,
    div[class*="st-emotion-cache"][class*="stAlert"] ,
    div[class*="st-emotion-cache"][data-testid*="stSuccess"],
    div[class*="st-emotion-cache"][data-testid*="stWarning"],
    div[class*="st-emotion-cache"][data-testid*="stInfo"],
    div[class*="st-emotion-cache"][data-testid*="stError"] {
        opacity: 1 !important;
        color: #FFFFFF !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.9);
        font-weight: 600;
    }

    /* Success (Green) */
    div[data-testid*="stSuccess"], div[class*="stSuccess"] {
        background: linear-gradient(90deg, #00C851, #007E33) !important;
        border: 2px solid #00FF7F !important;
        box-shadow: 0 0 20px rgba(0,255,127,0.7) !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }

    /* Info (Blue) */
    div[data-testid*="stInfo"], div[class*="stInfo"] {
        background: linear-gradient(90deg, #0099FF, #0056D2) !important;
        border: 2px solid #00BFFF !important;
        box-shadow: 0 0 20px rgba(0,191,255,0.8) !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }

    /* Warning (Gold) */
    div[data-testid*="stWarning"], div[class*="stWarning"] {
        background: linear-gradient(90deg, #FFD700, #FFB800) !important;
        color: #000000 !important;
        border: 2px solid #FFD700 !important;
        box-shadow: 0 0 20px rgba(255,215,0,0.8) !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }

    /* Error (Red) */
    div[data-testid*="stError"], div[class*="stError"] {
        background: linear-gradient(90deg, #FF4444, #CC0000) !important;
        border: 2px solid #FF6347 !important;
        box-shadow: 0 0 20px rgba(255,99,71,0.8) !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }

    /* Add slight glow + fade-in for all alerts */
    div[data-testid*="stAlert"] {
        animation: fadeIn 0.4s ease-in-out;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }



    /* ===== 🎨 IPL Themed Header Enhancements ===== */
    h1 {
        color: #FFD700 !important;
        text-shadow: 3px 3px 10px rgba(0,0,0,0.9),
                    0 0 20px rgba(255,215,0,0.8);
        font-weight: 900;
        letter-spacing: 1px;
    }
    h2, h3 {
        color: #FFFFFF !important;
        text-shadow: 1px 1px 6px rgba(0,0,0,0.9);
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    h4, h5, h6 {
        color: #FFFFFF !important;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.7);
        font-weight: 700;
    }

    /* Tabs */
    div[data-baseweb="tab-list"] {
        display: flex !important;
        justify-content: space-evenly !important;
        align-items: center !important;
        width: 100% !important;
        background: rgba(0, 0, 64, 0.5);
        border-radius: 10px;
        border: 2px solid rgba(255, 215, 0, 0.6);
        box-shadow: 0 0 10px rgba(255, 215, 0, 0.4);
        padding: 5px 0;
    }
    button[data-baseweb="tab"] {
        flex-grow: 1 !important;
        justify-content: center !important;
        text-align: center !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        padding: 10px 0 !important;
        background: transparent !important;
        border: none !important;
        border-radius: 8px 8px 0 0 !important;
        color: #FFD700 !important;
        transition: all 0.3s ease-in-out;
    }
    button[data-baseweb="tab"]:hover {
        background: rgba(255, 215, 0, 0.15) !important;
        transform: scale(1.05);
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(180deg, rgba(30,144,255,0.6), rgba(0,0,128,0.9)) !important;
        border-bottom: 3px solid #FFD700 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 10px rgba(255,215,0,0.4);
        transform: scale(1.08);
    }
    button[data-baseweb="tab"] > div {
        text-shadow: 1px 1px 4px rgba(0,0,0,0.8);
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #1E90FF, #00BFFF);
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 18em;
        font-weight: bold;
        font-size: 16px;
        border: none;
        box-shadow: 2px 2px 4px rgba(0,0,0,0.6);
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        color: black;
        transform: scale(1.05);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(0, 0, 64, 0.9);
        color: white;
        border-right: 2px solid #FFD700;
    }

    /* Tables */
    th {
        background-color: #1E90FF;
        color: #FFFFFF;
        font-weight: bold;
    }
    td {
        background-color: rgba(255, 255, 255, 0.95);
        color: #000000;
    }

    footer { visibility: hidden; }
                
    /* ===== 🩶 FIX FOR FADED MARKDOWN TEXT (e.g., Tactical Insights) ===== */
.stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span {
    color: #FFFFFF !important;
    opacity: 1.0 !important;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.9);
    font-weight: 600;
}

/* Slight glow to make list items pop */
.stMarkdown li::marker {
    color: #FFD700 !important; /* gold bullets/numbers */
    font-weight: bold;
}

    </style>
    """, unsafe_allow_html=True)

    # Show splash only once per session
if "splash_done" not in st.session_state:
    show_splash_screen()
    st.session_state.splash_done = True


    # -------------------- HEADER --------------------
st.markdown("""
    <div style='text-align:center; padding: 10px;'>
        <h1 style='color:#FFD700;'>Hybrid Cricket Intelligence Model (CIM)🏏 </h1>
        <h4 style='color:#E0E0E0;'>AI/ML-Powered Decision Support System for IPL Team Strategy</h4>
        <hr style='border:2px solid #FFD700; width:80%; margin:auto;'>
    </div>
""", unsafe_allow_html=True)


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
        df[col] = df[col].apply(lambda x: int(x) if pd.notnull(x) and float(x).is_integer()
                                else round(x,2) if pd.notnull(x) else x)
    return df

def clean_table(df):
    cols_to_drop = [c for c in df.columns if 'lower' in c or 'venue' in c or c == 'franchise']
    return df.drop(columns=cols_to_drop, errors='ignore')

def show_table(df, start_index=1):
    df = df.copy()
    df.index = range(start_index, start_index + len(df))
    st.dataframe(df, use_container_width=True)

# -------------------- INIT UI --------------------
add_bg_logo()
inject_css()

tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ GA Team Selection",
    "📊 SVM Player Performance",
    "🧩 Apriori Opponent Analysis",
    "🎯 XGBoost Matchup Prediction"
])

# -------------------- LOAD CORE DATA (cached) --------------------
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
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Best XI Selector (Genetic Algorithm)")

    input_team_display = st.selectbox("Select Franchise", franchise_display, key="ga_franchise")
    input_venue_display = st.selectbox("Select Venue", venue_display, key="ga_venue")
    st.session_state['input_venue'] = input_venue_display

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

    st.markdown("</div>", unsafe_allow_html=True)

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
                if any(x in r for x in ['bowler', 'spinner']):
                    return 'bowler'
                return 'batter'

            best_xi_df['group'] = best_xi_df['role'].apply(classify_group)
            preds_all = []

            # Batters
            batters = best_xi_df[best_xi_df['group']=='batter'].copy()
            if not batters.empty:
                Xb = batters[['runs','bat_avg','bat_sr']].copy()
                Xb = pd.concat([Xb, pd.get_dummies(batters[['venue']], drop_first=True)], axis=1)
                for col in feature_dict['batter']:
                    if col not in Xb.columns:
                        Xb[col] = 0
                Xb = Xb[feature_dict['batter']]
                batters['Predicted_Performance'] = batter_model.predict(Xb)
                preds_all.append(batters)

            # Bowlers
            bowlers = best_xi_df[best_xi_df['group']=='bowler'].copy()
            if not bowlers.empty:
                Xw = bowlers[['wickets','econ','runs']].copy()
                Xw = pd.concat([Xw, pd.get_dummies(bowlers[['venue']], drop_first=True)], axis=1)
                for col in feature_dict['bowler']:
                    if col not in Xw.columns:
                        Xw[col] = 0
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
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("🧩 Opponent Analysis- Matchup Insights (Apriori Algorithm)")
    st.info("Select opponent team and XI to find historical matchup patterns.")

    if st.session_state['best_xi'] is None:
        st.warning("Run the GA Team Selection (Tab 1) first.")
    else:
        df_roles_raw = roles_df
        franchise_display = sorted(df_roles_raw['franchise'].dropna().unique())
        opponent_team = st.selectbox("Select Opponent Franchise", franchise_display, key="apriori_opponent_team")
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

    st.markdown("</div>", unsafe_allow_html=True)

# =======================================================
# TAB 4: XGBOOST MATCHUP PREDICTION
# =======================================================
with tab4:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.subheader("🎯Matchup Dismissal Prediction (XGBoost Model)")
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
            players = []
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
                unknown=[]
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

        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown("#### 🏏 Your Playing XI")
            for p in my_xi_ordered:
                r = (best_xi_df.loc[best_xi_df['player_name']==p, 'role'].iloc[0]
                     if p in best_xi_df['player_name'].tolist() and 'role' in best_xi_df.columns else roles_map.get(p.lower(), ''))
                label = ROLE_LABEL.get(r, r.title() if r else '')
                st.write(f"- **{p}** {f'· {label}' if label else ''}")

            st.markdown("#### 🎯 Opponent XI")
            for p in opp_xi_ordered:
                r = roles_map.get(p.lower(), '')
                label = ROLE_LABEL.get(r, r.title() if r else '')
                st.write(f"- {p} {f'· {label}' if label else ''}")

            st.markdown("---")
            st.write(f"**Venue:** {venue}")
            phase_list = sorted(stats_df_local['phase'].dropna().unique())
            phase = st.selectbox("Select Phase", phase_list, index=0, key="xgb_phase")

            if st.button("Compute Matchups", key="xgb_compute"):
                st.session_state['_compute_matchups'] = True

        with col2:
            st.markdown("### 🔍 Inspect Player Matchups")
            inspect_side = st.radio("Choose XI", ("Your XI", "Opponent XI"), horizontal=True, key="xgb_inspect_side")
            if inspect_side == "Your XI":
                sel_player = st.selectbox("Select batsman from your XI", my_xi_ordered, key="xgb_sel_player")
                inspect_vs = opp_xi_ordered
            else:
                sel_player = st.selectbox("Select batsman from opponent XI", opp_xi_ordered, key="xgb_sel_player_opp")
                inspect_vs = my_xi_ordered

            st.markdown(f"Showing top bowlers vs **{sel_player}** at **{venue}** during **{phase}**.")

            if st.session_state.get('_compute_matchups', False):
                # NOTE: prefix model and encoders with underscores in the function signature so Streamlit will not attempt to hash them.
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

                with st.spinner("Calculating probabilities..."):
                    pair_probs = compute_probs_for_pair(sel_player, inspect_vs, venue, phase, stats_df_local, _model=xgb_model, _encoders=xgb_encoders, feat_cols=xgb_feat_cols)
                    df_top = pd.DataFrame(pair_probs, columns=["Bowler","Dismissal %"])
                    df_top["Dismissal %"] = df_top["Dismissal %"].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "–")
                    st.session_state['xgb_matchups'] = df_top

                top_n = 6
                st.markdown(f"**Top {top_n} bowlers vs {sel_player}**")
                st.table(st.session_state['xgb_matchups'].head(top_n))

                with st.expander("Show full matchups for this player"):
                    st.dataframe(st.session_state['xgb_matchups'], use_container_width=True)
            else:
                st.info("Click **Compute Matchups** (left) to generate predictions; then select a player to inspect.")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("""
<hr>
<p style='text-align:center; color:white'>© 2025 Hybrid Cricket Intelligence Model (CIM) | Developed by Santhoji V</p>
""", unsafe_allow_html=True)
