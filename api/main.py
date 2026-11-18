"""
FastAPI backend.

Endpoints
---------
GET  /health        -> liveness + which model is loaded
GET  /model-info    -> model details + comparison table (if trained)
POST /predict       -> label, risk score, highlighted words, red-flag list

Run locally:
    uvicorn api.main:app --reload --port 8000
Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Make `src` importable when running `uvicorn api.main:app` from the repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import __version__  # noqa: E402
from src.config import RESULTS_CSV  # noqa: E402
from src.predict import get_predictor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

@asynccontextmanager
async def lifespan(_app):
    get_predictor()  # load models once at startup, not on the first request
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Fake Job & Internship Scam Detector",
    description="Predicts whether a job/internship post is a scam and explains why.",
    version=__version__,
)

# The Chrome extension and Streamlit call this API from other origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    text: str = Field(..., max_length=20_000, description="Full job / internship post")
    company: Optional[str] = Field("", max_length=300)
    email: Optional[str] = Field("", max_length=300)
    salary: Optional[str] = Field("", max_length=300)
    explain: Literal["auto", "lime", "none"] = "auto"

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Work from home! Earn ₹50,000 per week. Pay registration fee ₹999. "
                        "WhatsApp HR on 9000000001. No interview, direct joining!",
                "company": "",
                "email": "hr.jobs@gmail.com",
                "salary": "₹50,000 per week",
            }
        }
    }


class Term(BaseModel):
    term: str
    weight: float


class Segment(BaseModel):
    text: str
    score: float
    kind: Optional[str] = None


class RedFlagItem(BaseModel):
    id: str
    name: str
    description: str
    weight: float
    triggered: bool
    evidence: list[str]


class PredictResponse(BaseModel):
    label: Literal["Real", "Fake"]
    is_fake: bool
    risk_score: float = Field(..., ge=0, le=100)
    risk_level: Literal["low", "medium", "high"]
    probability: float
    model_probability: Optional[float]
    rule_score: float
    threshold: float
    model_name: str
    explanation_method: str
    highlighted_words: list[Term]
    trust_words: list[Term]
    highlighted_segments: list[Segment]
    red_flags: list[RedFlagItem]
    triggered_flags: int
    language: str
    low_confidence: bool
    warnings: list[str]


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, **get_predictor().info()}


@app.get("/model-info")
def model_info() -> dict:
    info = get_predictor().info()
    if RESULTS_CSV.exists():
        info["results"] = pd.read_csv(RESULTS_CSV).fillna(0).to_dict(orient="records")
    else:
        info["results"] = []
    return info


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> dict:
    try:
        return get_predictor().predict(
            req.text, req.company or "", req.email or "", req.salary or "", explain=req.explain
        )
    except ValueError as exc:  # e.g. empty text after stripping
        raise HTTPException(status_code=422, detail=str(exc)) from exc
