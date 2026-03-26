from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent

# Model file paths
MODEL_PATH      = ROOT / "model/ensemble_model.pkl"
ENCODERS_PATH   = ROOT / "model/label_encoders.pkl"
FEATURES_PATH   = ROOT / "model/feature_names.json"
METRICS_PATH    = ROOT / "model/model_metrics.json"

# Prediction bounds
MIN_PROBABILITY = 0.05
MAX_PROBABILITY = 0.95

# API settings
API_VERSION     = "1.0.0"
API_TITLE       = "IPL Probability Engine"
API_DESCRIPTION = """
ML-powered IPL match win probability API.
Trained on IPL 2008-2024, T20 Internationals
and BBL ball-by-ball data.
Powers ipl-bet in production.
"""

# Known IPL 2026 teams
IPL_TEAMS = [
    "Mumbai Indians",
    "Chennai Super Kings",
    "Royal Challengers Bengaluru",
    "Kolkata Knight Riders",
    "Sunrisers Hyderabad",
    "Delhi Capitals",
    "Punjab Kings",
    "Rajasthan Royals",
    "Gujarat Titans",
    "Lucknow Super Giants"
]

# Known IPL venues
IPL_VENUES = [
    "Mumbai",
    "Chennai",
    "Bengaluru",
    "Kolkata",
    "Hyderabad",
    "Delhi",
    "Ahmedabad",
    "Jaipur",
    "Lucknow",
    "Mohali"
]
