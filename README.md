# Hybrid Cricket Intelligence Model (CIM)

A Streamlit-based AI/ML toolkit for IPL team strategy — providing Best XI selection (Genetic Algorithm), player performance predictions (SVM), opponent matchup insights (Apriori), and bowler-batsman dismissal probability (XGBoost).

---

## Quick Overview

- Language: Python
- UI: Streamlit
- Main features:
  - GA Team Selection: generate a Best XI for a franchise and venue
  - SVM Player Performance: predict player performance categories
  - Apriori Opponent Analysis: find historical matchup patterns
  - XGBoost Matchup Prediction: compute bowler-vs-batsman dismissal probabilities

This repository contains the code, data references, images, and pre-trained model artifacts required to run a local interactive dashboard.

---

## Repository Layout (important files)

- `CIM/`
  - `models/` - The main Streamlit app and model scripts
    - `app.py` - Main Streamlit application (tabs: GA, SVM, Apriori, XGBoost)
    - `ga_team_selector.py` - Genetic algorithm implementation (Best XI)
    - `ap_opponent_analysis.py` - Apriori-based opponent analysis helper
    - `svm_player_performance.py` - SVM model code and helper functions
    - `train_xgb_delivery.py` - XGBoost training utilities (if present)
    - `player_roles_cleaned.csv` - cleaned roles reference
  - `data/` - CSV data used by the app
    - `player_stats_venue.csv`
    - `player_roles.csv`
    - `ball_by_ball_stats_ap.csv`
    - `match_stats.csv` and others used for analyses
  - `pictures/`
    - `logos/` - team logo images (referenced by `app.py`)
    - `ipl_stadium_bg.jpg` - background image
    - `ipl_logo.png` - app header logo
  - `static/` - static assets used by the UI

Top-level CSV(s):
- `player_stats_venue.csv` (also present at repository root in some copies)

---

## Prerequisites

- Python 3.8+ recommended
- Install required packages (example):

```powershell
pip install streamlit pandas numpy scikit-learn xgboost joblib matplotlib seaborn
```

(If you have a `requirements.txt` file, prefer `pip install -r requirements.txt`.)

---

## Configuration

`app.py` contains a number of hard-coded paths near the top. Default values used in the app:

```python
STATS_FILE = "D:/AI ML Cricket Project CIM model/CIM/data/player_stats_venue.csv"
ROLES_FILE = "D:/AI ML Cricket Project CIM model/CIM/data/player_roles.csv"
BALL_BY_BALL_FILE = "D:/AI ML Cricket Project CIM model/CIM/data/ball_by_ball_stats_ap.csv"
BG_PATH = "D:/AI ML Cricket Project CIM model/CIM/pictures/ipl_stadium_bg.jpg"
LOGO_PATH = "D:/AI ML Cricket Project CIM model/CIM/pictures/ipl_logo.png"
LOGOS_PATH = "D:/AI ML Cricket Project CIM model/CIM/pictures/logos"

# Model artifact names (expected in models/ working dir)
SVM_BATTER = "svm_batter_model.pkl"
SVM_BOWLER = "svm_bowler_model.pkl"
SVM_FEAT = "svm_player_performance_features.pkl"
XGB_MODEL = "xgb_delivery_model.pkl"
XGB_ENC = "xgb_delivery_label_encoders.pkl"
XGB_FEAT = "xgb_delivery_features.pkl"
```

If your working directories differ, update these constants at the top of `app.py` before running.

---

## Running the App

1. Open a terminal and change to the `CIM/models` directory:

```powershell
cd "D:\AI ML Cricket Project CIM model\CIM\models"
```

2. Run Streamlit:

```powershell
python -m streamlit run app.py
```

3. The app will open in your browser (or use the local URL shown in the terminal). Navigate between tabs to run:
- Tab 1: GA Team Selection — select Franchise + Venue, click "Generate Best XI"
- Tab 2: SVM Player Performance — compute player performance predictions for the Best XI
- Tab 3: Apriori Opponent Analysis — select opponent and XI to run Apriori
- Tab 4: XGBoost Matchup Prediction — compute bowler-batsman dismissal probabilities (relies on Best XI + Opponent & Venue)

Notes:
- The Apriori opponent selection is persisted to session state and used by the XGBoost tab to display the opponent badge.
- Some tabs require that the GA (Tab 1) has been run first (Best XI available in session).

---

## Data & Model Files

- Data CSVs must be present at the paths configured in `app.py`. If data files are large, place them in the `CIM/data/` folder and update the constants.
- Pre-trained model artifacts (SVM/XGBoost) must be present in `CIM/models/` or update the path variables. If you need to re-train models, use the training scripts in `CIM/models/` if available.

---

## UI & Styling

- `app.py` injects a custom CSS block via `inject_css()` for a premium dashboard look. If you need to revert to simple styling, remove or edit that function.
- Team logos are loaded from `LOGOS_PATH`. Team-to-logo mapping is defined in `app.py` via the `TEAM_LOGO_MAP` dictionary.
- The app uses Streamlit session state keys (e.g. `best_xi`, `input_team_display`, `opponent_team`, `xgb_matchups`) to pass information across tabs.

---

## Troubleshooting

- Common errors:
  - File not found: verify the CSV and image paths in `app.py`.
  - Model loading error: ensure `.pkl` files exist and are compatible with your Python package versions.
  - Streamlit caching issues: try restarting the app and clearing Streamlit cache (`streamlit cache clear` for older versions or delete `.streamlit` cache folder).

- If the UI shows raw `</div>` text or other stray HTML: this was caused previously by manually writing closing tags via `st.markdown("</div>", unsafe_allow_html=True)`. The app has been updated to avoid rendering stray closing tags; if you see these, ensure you are running the latest `app.py`.

---

## Developer Notes

- To add a new team logo: copy an image into `CIM/pictures/logos/` and add the filename to the `TEAM_LOGO_MAP` dictionary in `app.py`.
- Session keys of interest:
  - `st.session_state['best_xi']` — pandas DataFrame of best XI
  - `st.session_state['input_team_display']` — the franchise selected in GA
  - `st.session_state['opponent_team']` — selected opponent from Apriori tab (used by XGBoost)
  - `st.session_state['xgb_matchups']` — latest XGBoost matchup DataFrame

---

## Contribution

- Fork and create feature branches for changes.
- Keep UI and CSS changes isolated, and avoid embedding raw closing HTML tags via `st.markdown` (Streamlit sometimes renders them as text).

---

## Contact

Maintainer: Santhoji V

If you'd like help running the app or customizing visuals, open an issue or contact the maintainer directly.

---

*Generated README — adapt paths and instructions to your local environment as needed.*
