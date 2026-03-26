from pydantic import BaseModel, field_validator


class MatchState(BaseModel):
    # Required fields
    batting_team:          str
    bowling_team:          str
    city:                  str
    runs_left:             float
    balls_left:            float
    wickets_remaining:     float
    crr:                   float
    rrr:                   float

    # Optional with smart defaults
    league:                str   = "IPL"
    over_number:           int   = 10
    total_extras:          float = 0.0
    extras_rate:           float = 0.0
    wides_this_innings:    float = 0.0
    no_balls_this_innings: float = 0.0
    boundary_percentage:   float = 0.25
    dot_ball_percentage:   float = 0.30
    partnership_runs:      float = 20.0
    partnership_balls:     float = 18.0
    recent_12_balls_rr:    float = 0.0
    last_3_overs_avg:      float = 0.0

    @field_validator('runs_left')
    @classmethod
    def runs_left_valid(cls, v):
        if v < 0:
            raise ValueError('runs_left cannot be negative')
        return v

    @field_validator('balls_left')
    @classmethod
    def balls_left_valid(cls, v):
        if not 0 <= v <= 120:
            raise ValueError('balls_left must be 0-120')
        return v

    @field_validator('wickets_remaining')
    @classmethod
    def wickets_valid(cls, v):
        if not 0 <= v <= 10:
            raise ValueError('wickets_remaining must be 0-10')
        return v

    @field_validator('crr')
    @classmethod
    def crr_valid(cls, v):
        if v < 0:
            raise ValueError('crr cannot be negative')
        return round(v, 3)

    @field_validator('rrr')
    @classmethod
    def rrr_valid(cls, v):
        if v < 0:
            raise ValueError('rrr cannot be negative')
        return round(v, 3)


class ProbabilityRange(BaseModel):
    low:  float
    high: float


class PredictionResponse(BaseModel):
    batting_team_win_prob:  float
    bowling_team_win_prob:  float
    confidence:             str
    confidence_score:       float
    probability_range:      ProbabilityRange
    model_accuracy:         float
    brier_score:            float
    source:                 str


class HealthResponse(BaseModel):
    status:        str
    model:         str
    accuracy:      float
    brier_score:   float
    training_date: str
    api_version:   str
