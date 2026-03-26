from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from api.config import (
    API_TITLE, API_DESCRIPTION,
    API_VERSION, IPL_TEAMS, IPL_VENUES
)
from api.schemas import (
    MatchState, PredictionResponse, HealthResponse
)
from api.predictor import IPLPredictor

# Global predictor instance
predictor: IPLPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global predictor
    predictor = IPLPredictor()
    yield
    # Cleanup on shutdown (if needed)


# Create FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Health check — returns model info"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    m = predictor.metrics
    return HealthResponse(
        status="ok",
        model=m.get('model_type', 'Ensemble'),
        accuracy=m['overall_accuracy'],
        brier_score=m['brier_score'],
        training_date=m.get('training_date', 'unknown'),
        api_version=API_VERSION
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(state: MatchState):
    """
    Predict win probability for a live IPL match situation.

    Returns batting team win probability, confidence level,
    and probability range.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = predictor.predict(state)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/model-info", tags=["Info"])
async def model_info():
    """Full model metrics and training info"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor.metrics


@app.get("/teams", tags=["Info"])
async def get_teams():
    """List all IPL 2026 teams"""
    return {"teams": IPL_TEAMS, "count": len(IPL_TEAMS)}


@app.get("/venues", tags=["Info"])
async def get_venues():
    """List all IPL venues"""
    return {"venues": IPL_VENUES, "count": len(IPL_VENUES)}


@app.get("/example", tags=["Info"])
async def example():
    """Example prediction request"""
    return {
        "description": (
            "CSK chasing 175 vs MI at Chennai. "
            "Over 13, score 108/3, need 67 off 42 balls."
        ),
        "request": {
            "batting_team":          "Chennai Super Kings",
            "bowling_team":          "Mumbai Indians",
            "city":                  "Chennai",
            "runs_left":             67,
            "balls_left":            42,
            "wickets_remaining":     7,
            "crr":                   8.31,
            "rrr":                   9.57,
            "over_number":           13,
            "total_extras":          8,
            "extras_rate":           0.62,
            "boundary_percentage":   0.28,
            "dot_ball_percentage":   0.31,
            "partnership_runs":      34,
            "partnership_balls":     28,
            "recent_12_balls_rr":    7.5,
            "last_3_overs_avg":      8.2
        },
        "expected_response": {
            "batting_team_win_prob": "~0.42",
            "bowling_team_win_prob": "~0.58",
            "confidence":            "High",
            "source":                "ml-ensemble"
        }
    }


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name":    "IPL Probability Engine",
        "version": API_VERSION,
        "docs":    "/docs",
        "health":  "/health",
        "predict": "/predict"
    }


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
