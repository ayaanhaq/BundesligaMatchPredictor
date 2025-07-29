import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# --- Load Data ---
data_path = os.path.join(os.path.dirname(__file__), 'match_data_expanded.csv')
matches = pd.read_csv(data_path, index_col=0)
matches["Date"] = pd.to_datetime(matches["Date"], errors='coerce')

# --- Name normalization mapping ---
name_normalize = {
    'Köln': 'Koln',
    'Darmstadt 98': 'Darmstadt',
    'Eint Frankfurt': 'Eintracht Frankfurt',
    'Gladbach': 'Monchengladbach',
    'Holstien Kiel': 'Holstein Kiel',
}
def normalize_name(name):
    return name_normalize.get(name, name)

# Normalize Team and Opponent columns
matches['Team'] = matches['Team'].apply(normalize_name)
matches['Opponent'] = matches['Opponent'].apply(normalize_name)

# --- Preprocessing (as in notebook) ---
matches["Venue_code"] = matches["Venue"].astype("category").cat.codes
matches["Opp_code"] = matches["Opponent"].astype("category").cat.codes
matches["hour"] = matches["Time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["Date"].dt.day_of_week
matches["target"] = (matches["Result"] == "W").astype("int")

# --- Rolling averages ---
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("Date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

cols = ["GF", "GA", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]
new_cols = [f"{c}_rolling" for c in cols]

matches_rolling = matches.groupby("Team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel("Team")
matches_rolling.index = range(matches_rolling.shape[0])

# --- Model ---
predictors = ["Venue_code", "Opp_code", "hour", "day_code"] + new_cols
train = matches_rolling[matches_rolling["Date"] < "2025-01-01"]
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
rf.fit(train[predictors], train["target"])

# --- Team/Logo Setup ---
logo_dir = os.path.join(os.path.dirname(__file__), 'logos')
team_names = sorted(matches["Team"].unique())
# Map team names to logo filenames (handle .svg/.png)
logo_map = {}
for file in os.listdir(logo_dir):
    if file.endswith('.svg') or file.endswith('.png'):
        name = file.replace('.svg', '').replace('.png', '').replace('Holstien', 'Holstein')
        logo_map[normalize_name(name).lower()] = file

def get_logo(team):
    key = normalize_name(team).lower()
    file = logo_map.get(key)
    if file:
        return os.path.join(logo_dir, file)
    return None

# --- Streamlit UI ---
st.set_page_config(page_title="Bundesliga Match Predictor", layout="centered")

# Bundesliga logo in title
bundesliga_logo_path = os.path.join(logo_dir, 'Bundesliga.svg')
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image(bundesliga_logo_path, width=70)
with col_title:
    st.markdown("<h1 style='display:inline; vertical-align:middle;'>Bundesliga Match Predictor</h1>", unsafe_allow_html=True)
st.write("Predict the outcome of a Bundesliga match using historical data and rolling averages.")

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Select Home Team", team_names, key="team1")
    logo1 = get_logo(team1)
    if logo1:
        st.image(logo1, width=120)
with col2:
    team2 = st.selectbox("Select Away Team", [t for t in team_names if t != team1], key="team2")
    logo2 = get_logo(team2)
    if logo2:
        st.image(logo2, width=120)

if team1 == team2:
    st.warning("Please select two different teams.")
    st.stop()

# --- Prediction Logic ---
st.subheader(f"Prediction: {team1} vs {team2}")

def get_latest_features(team, opponent, venue):
    team_matches = matches_rolling[(matches_rolling["Team"] == team)]
    if team_matches.empty:
        return None
    last_match = team_matches.iloc[-1]
    features = last_match[predictors].copy()
    features["Venue_code"] = matches_rolling["Venue"].astype("category").cat.categories.get_loc(venue)
    opp_cats = matches_rolling["Opponent"].astype("category").cat.categories
    norm_opponent = opponent
    if norm_opponent not in opp_cats:
        for cat in opp_cats:
            if cat.lower() == norm_opponent.lower():
                norm_opponent = cat
                break
    opp_code = matches_rolling["Opponent"].astype("category").cat.categories.get_loc(norm_opponent)
    features["Opp_code"] = opp_code
    return features

venue = 'Home'
features = get_latest_features(team1, team2, venue)
if features is not None:
    X = pd.DataFrame([features])
    pred = rf.predict(X)[0]
    proba = rf.predict_proba(X)[0]
    st.markdown(f"**Prediction:** {'Win' if pred == 1 else 'Not Win'} for {team1}")
    st.progress(float(proba[1]), text=f"Win probability: {proba[1]*100:.1f}%")

    # --- Eye Candy: Recent Form and Stats ---
    st.markdown("---")
    st.markdown("### Recent Form & Stats")
    form_cols = st.columns(2)
    def get_recent_form(team, n=5):
        team_matches = matches_rolling[matches_rolling["Team"] == team].sort_values("Date", ascending=False).head(n)
        return team_matches["Result"].tolist()
    def get_rolling_stats(team):
        team_matches = matches_rolling[matches_rolling["Team"] == team]
        if team_matches.empty:
            return {}
        last = team_matches.iloc[-1]
        return {k.replace('_rolling',''): last[k] for k in new_cols}

    # Prepare stats for both teams
    stats1 = get_rolling_stats(team1)
    stats2 = get_rolling_stats(team2)
    stat_names = [k.replace('_', ' ').capitalize() for k in stats1.keys()]
    # Home team
    with form_cols[0]:
        st.markdown(f"**{team1}**")
        form1 = get_recent_form(team1)
        st.markdown("Recent form: " + ' '.join([f"<span style='color:{'green' if r=='W' else 'orange' if r=='D' else 'red'}'>{r}</span>" for r in form1]), unsafe_allow_html=True)
    # Away team
    with form_cols[1]:
        st.markdown(f"**{team2}**")
        form2 = get_recent_form(team2)
        st.markdown("Recent form: " + ' '.join([f"<span style='color:{'green' if r=='W' else 'orange' if r=='D' else 'red'}'>{r}</span>" for r in form2]), unsafe_allow_html=True)

    # --- Rolling stats table ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Rolling Stats (Previous 3 Matches)")
    stats_table = f"""
    <style>
    .stats-card {{
        width: 100%;
        border-radius: 18px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        background: #f5f6fa;
        padding: 18px 0 10px 0;
        margin-bottom: 1.5em;
        border: none;
    }}
    .stats-card table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 8px;
        background: transparent;
    }}
    .stats-card th, .stats-card td {{
        padding: 10px 18px;
        text-align: center;
        border: none;
        font-size: 1.08em;
    }}
    .stats-card th {{
        background: transparent;
        color: #d10214;
        font-size: 1.1em;
        font-weight: 700;
        border: none;
    }}
    .stats-card td.stat-name {{
        text-align: left;
        font-weight: 500;
        color: #222;
        border-radius: 12px 0 0 12px;
        background: #f3f4f6;
    }}
    .stats-card td.stat-val1 {{
        font-size: 1.18em;
        font-weight: 600;
        background: #e3f0ff;
        border-radius: 0 0 0 12px;
        color: #1a3a5d;
    }}
    .stats-card td.stat-val2 {{
        font-size: 1.18em;
        font-weight: 600;
        background: #ffeaea;
        border-radius: 0 12px 12px 0;
        color: #a11a1a;
    }}
    </style>
    <div class='stats-card'>
    <table>
      <tr>
        <th>Stat</th>
        <th>{team1}</th>
        <th>{team2}</th>
      </tr>
    """
    for i, stat in enumerate(stat_names):
        val1 = stats1[list(stats1.keys())[i]] if stats1 else '-'
        val2 = stats2[list(stats2.keys())[i]] if stats2 else '-'
        stats_table += f"<tr><td class='stat-name'>{stat}</td><td class='stat-val1'>{val1:.2f}</td><td class='stat-val2'>{val2:.2f}</td></tr>"
    stats_table += "</table></div>"
    st.markdown(stats_table, unsafe_allow_html=True)
else:
    st.error("Insufficient data for prediction.")

st.caption("Model uses rolling averages of recent games and match context for prediction. Logos © respective clubs.") 