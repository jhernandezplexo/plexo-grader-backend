# main.py
import os
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ------------------------------
# App + CORS (open for now)
# ------------------------------
app = FastAPI(title="Plexo Grader Backend", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Env
# ------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
PAGESPEED_API_KEY = os.getenv("PAGESPEED_API_KEY", "").strip()

# ------------------------------
# Models (minimal for UI)
# ------------------------------
class ScoreComponent(BaseModel):
    key: str
    weight: float
    raw: float
    points: float
    explain: str

class ScoreResponse(BaseModel):
    score: int
    components: list[ScoreComponent] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

class LeadIn(BaseModel):
    name: str
    email: str
    business_name: str
    website: Optional[str] = None
    location: Optional[str] = None
    target_category: Optional[str] = None

# ------------------------------
# Health
# ------------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"ok": "true"}

# ------------------------------
# Helpers (stubs you can expand)
# ------------------------------
def _ok(v: Optional[str]) -> bool:
    return bool(v and v.strip())

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

# Example bayesian smoothing you wanted
def bayesian_rating(rating: float, reviews: int, m: int = 200, C: float = 4.3) -> float:
    """
    bayes = (m*C + n*R) / (m+n)
    """
    n = max(0, reviews)
    return (m * C + n * rating) / (m + n)

# ------------------------------
# Stub /score endpoint
# ------------------------------
@app.post("/score", response_model=ScoreResponse)
async def score(payload: LeadIn) -> ScoreResponse:
    """
    Temporary/stub scorer so the UI does not 404.
    Replace internals with the real logic when your keys & Google calls are ready.
    """
    # Very light validation
    warnings: list[str] = []
    if not _ok(payload.business_name):
        warnings.append("Missing business name.")
    if payload.website and not payload.website.startswith(("http://", "https://")):
        warnings.append("Website should start with http:// or https://")

    # Demo components using your desired weights:
    # performance: 30%, gbp_quality: 45%, category_match: 10%, competition_gap: 15%
    components: list[ScoreComponent] = []

    # --- performance (stub) ---
    perf_raw = 0.50  # pretend PageSpeed normalized 0..1
    components.append(
        ScoreComponent(
            key="performance",
            weight=0.30,
            raw=perf_raw,
            points=round(perf_raw * 30, 1),
            explain="PageSpeed (placeholder)",
        )
    )

    # --- gbp_quality (stub) ---
    gbp_raw = 0.95  # pretend normalized 0..1
    components.append(
        ScoreComponent(
            key="gbp_quality",
            weight=0.45,
            raw=gbp_raw,
            points=round(gbp_raw * 45, 1),
            explain="GBP rating/reviews/presence (placeholder)",
        )
    )

    # --- category_match (stub) ---
    cat_raw = 1.0 if _ok(payload.target_category) else 0.0
    components.append(
        ScoreComponent(
            key="category_match",
            weight=0.10,
            raw=cat_raw,
            points=round(cat_raw * 10, 1),
            explain="GBP types vs target_category (placeholder)",
        )
    )

    # --- competition_gap (stub) ---
    gap_raw = 0.90
    components.append(
        ScoreComponent(
            key="competition_gap",
            weight=0.15,
            raw=gap_raw,
            points=round(gap_raw * 15, 1),
            explain="Your rating vs nearby competitors (placeholder)",
        )
    )

    total_points = sum(c.points for c in components)
    final_score = int(round(total_points))  # 0..100 style

    return ScoreResponse(score=final_score, components=components, warnings=warnings)
