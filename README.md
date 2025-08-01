
# Soccer Match Predictor

A supervised machine learning model which predicts the outcomes of future soccer matches using in-game statistics data obtained from historical games

This machine learning model leverages over 600 games played over the course of the last 2 seasons in the German football league to make accurate predictions about future match-ups.

Check out the project right [here](https://match-predictor.streamlit.app/)

![Match Predictor](https://github.com/ayaanhaq/BundesligaMatchPredictor/blob/deployment/logos/ss1.png "Match Predictor")

Historical statistics such as Shots, Shots on Target, Free Kicks, Penalties, etc. are taken into account when making predictions. Furthermore, team form is also incorporated into the calculations, as well as rolling averages of each statistic from the last 3 games to better reflect the momentum each team is carrying into the next match. 

![Match Predictor](https://github.com/ayaanhaq/BundesligaMatchPredictor/blob/deployment/logos/ss2.png "Match Predictor")

## Features
- Match Outcome Prediction: Predicts Bundesliga match results using a trained Random Forest classifier on 600+ historical games.
- Stat-Driven Insights: Incorporates advanced match stats such as shots, shots on target, avg shot distance, penalties, and free kicks.
- Rolling Averages: Factors in recent team form using rolling statistical windows for more accurate, context-aware predictions.
- Interactive Web App: Built with Streamlit, providing real-time predictions with user-friendly UI and confidence scores.
- Deployed and Live: Instantly accessible via web, no setup needed to try it out.

## Technologies Used
- Python - Language used for data processing and modeling
- Pandas & NumPy – For efficient data manipulation, cleaning, and rolling average calculations
- Scikit-learn – To train and evaluate the Random Forest model for match outcome prediction
- Streamlit – Interactive web app for real-time predictions and user interaction
- Matplotlib & Seaborn – Visualize model performance and feature importance

## Contribution

Feel free to contribute to this project, or maybe even expand on it further.

