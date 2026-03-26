import pickle
import json
import numpy as np
import pandas as pd
from api.config import (
    MODEL_PATH, ENCODERS_PATH,
    FEATURES_PATH, METRICS_PATH,
    MIN_PROBABILITY, MAX_PROBABILITY
)


class IPLPredictor:
    def __init__(self):
        self.model        = None
        self.encoders     = None
        self.feature_cols = None
        self.metrics      = None
        self._load_all()

    def _load_all(self):
        print("Loading ML model...")

        # Load ensemble model
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)

        # Load label encoders
        with open(ENCODERS_PATH, 'rb') as f:
            self.encoders = pickle.load(f)

        # Load feature names
        with open(FEATURES_PATH) as f:
            self.feature_cols = json.load(f)

        # Load metrics
        with open(METRICS_PATH) as f:
            self.metrics = json.load(f)

        print(f"✅ Model loaded successfully")
        print(f"   Accuracy: {self.metrics['overall_accuracy']:.3f}")
        print(f"   Brier:    {self.metrics['brier_score']:.3f}")

    def _encode(self, value, key):
        """Encode categorical with fallback"""
        encoder = self.encoders[key]
        val_str = str(value)
        if val_str in encoder.classes_:
            return int(encoder.transform([val_str])[0])
        # Unknown value → use most common
        return 0

    def _build_features(self, state) -> pd.DataFrame:
        """Build feature DataFrame from match state"""

        # Derived features
        run_rate_diff     = state.crr - state.rrr
        pressure_index    = state.rrr / max(state.crr, 0.1)
        required_balls    = state.balls_left / max(state.runs_left, 1)
        wickets_fallen    = 10 - state.wickets_remaining
        wickets_pressure  = wickets_fallen / max(state.over_number, 1)
        momentum_vs_req   = (state.recent_12_balls_rr - state.rrr)
        extra_balls       = (state.wides_this_innings + state.no_balls_this_innings)
        partnership_rr    = (
            state.partnership_runs /
            max(state.partnership_balls / 6, 0.1)
        )

        features = {
            # Tier 1: Core
            'runs_left':          state.runs_left,
            'balls_left':         state.balls_left,
            'wickets_remaining':  state.wickets_remaining,
            'crr':                state.crr,
            'rrr':                state.rrr,
            'run_rate_diff':      round(run_rate_diff, 3),

            # Tier 2: Context
            'batting_team': self._encode(state.batting_team, 'batting_team'),
            'bowling_team': self._encode(state.bowling_team, 'bowling_team'),
            'city':         self._encode(state.city, 'city'),
            'league':       self._encode(state.league, 'league'),

            # Tier 3: Phase
            'over_number':       state.over_number,
            'is_powerplay':      int(state.over_number <= 6),
            'is_middle_overs':   int(7 <= state.over_number <= 15),
            'is_death_overs':    int(state.over_number >= 16),

            # Tier 4: Pressure
            'pressure_index':         round(pressure_index, 3),
            'required_balls_per_run': round(required_balls, 3),
            'wickets_pressure':       round(wickets_pressure, 3),

            # Tier 5: Momentum
            'recent_12_balls_rr':   state.recent_12_balls_rr,
            'last_3_overs_avg':     state.last_3_overs_avg,
            'momentum_vs_required': round(momentum_vs_req, 3),

            # Tier 6: Extras
            'total_extras':       state.total_extras,
            'extras_rate':        state.extras_rate,
            'wides_this_innings': state.wides_this_innings,
            'extra_balls_gained': extra_balls,

            # Tier 7: Aggression
            'boundary_percentage': state.boundary_percentage,
            'dot_ball_percentage': state.dot_ball_percentage,

            # Tier 8: Partnership
            'partnership_runs':   state.partnership_runs,
            'partnership_balls':  state.partnership_balls,
        }

        # Build DataFrame in exact feature order
        df = pd.DataFrame([features])

        # Only keep features model was trained on
        available = [f for f in self.feature_cols if f in df.columns]
        df = df[available]

        # Fill any missing features with 0
        for col in self.feature_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[self.feature_cols]
        return df

    def _get_confidence(self, prob: float, balls_left: float) -> tuple:
        """Calculate confidence level"""
        extremity   = abs(prob - 0.5) * 2
        overs_factor = balls_left / 120
        score = (extremity * 0.6 + (1 - overs_factor) * 0.4)
        score = round(float(score), 3)

        if score > 0.75:
            label = "Very High"
        elif score > 0.55:
            label = "High"
        elif score > 0.35:
            label = "Medium"
        else:
            label = "Low"

        return label, score

    def predict(self, state) -> dict:
        """Generate win probability prediction"""

        # Build features
        features = self._build_features(state)

        # Get ensemble prediction
        ensemble = self.model
        if isinstance(ensemble, dict) and 'models' in ensemble:
            # Weighted ensemble
            probs_list = [
                m.predict_proba(features)[0][1]
                for m in ensemble['models'].values()
            ]
            weights = ensemble['weights']
            total_w = sum(weights)
            prob = float(sum(w * p for w, p in zip(weights, probs_list)) / total_w)
        else:
            # Single model
            prob = float(ensemble.predict_proba(features)[0][1])

        # Clamp probability
        prob = max(MIN_PROBABILITY, min(MAX_PROBABILITY, prob))
        prob = round(prob, 4)

        # Confidence
        confidence, conf_score = self._get_confidence(prob, state.balls_left)

        # Uncertainty range
        uncertainty = (1 - conf_score) * 0.08
        prob_low  = round(max(MIN_PROBABILITY, prob - uncertainty), 4)
        prob_high = round(min(MAX_PROBABILITY, prob + uncertainty), 4)

        return {
            'batting_team_win_prob': prob,
            'bowling_team_win_prob': round(1 - prob, 4),
            'confidence':            confidence,
            'confidence_score':      conf_score,
            'probability_range': {
                'low':  prob_low,
                'high': prob_high
            },
            'model_accuracy': float(self.metrics['overall_accuracy']),
            'brier_score':    float(self.metrics['brier_score']),
            'source':         'ml-ensemble'
        }
