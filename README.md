# Bundesliga Match Predictor

This project includes a Streamlit app to predict Bundesliga match outcomes using a machine learning model trained on historical data and rolling averages.

## Setup

1. Install dependencies:

```bash
pip install -r Bundesliga\ ML/requirements.txt
```

2. Ensure the following files are present in the `Bundesliga ML` directory:
   - `match_data_expanded.csv`
   - `logos/` folder with team logos
   - `streamlit_app.py`

## Running the App

From the `Bundesliga ML` directory, run:

```bash
streamlit run streamlit_app.py
```

The app will open in your browser. Select two different teams to see the predicted outcome and win probability, with team logos displayed.

---

*Model uses rolling averages of recent games and match context for prediction. Logos © respective clubs.*
