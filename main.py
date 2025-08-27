from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

app = FastAPI(title="Plexo Grader Backend (Minimal OK Boot)")

# ---------- Models (simple) ----------

class LeadIn(BaseModel):
    name: str
    email: str
    business_name: str
    website: Optional[str] = None

class LeadOut(BaseModel):
    id: str
    status: str = "queued"

class BusinessOut(BaseModel):
    found: bool
    name: Optional[str] = None
    formatted_address: Optional[str] = None
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    website: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    open_now: Optional[bool] = None
    competitors: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None

class ScoreComponent(BaseModel):
    key: str
    weight: float
    raw: Optional[float] = None
    points: Optional[float] = None
    explain: str

class ScoreOut(BaseModel):
    input: Dict[str, Any]
    pagespeed: Dict[str, Any]
    gbp: Dict[str, Any]
    components: List[ScoreComponent]
    score: int
    warnings: Optional[List[str]] = None


# ---------- Health ----------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"ok": "true"}


# ---------- Leads (stub) ----------

@app.post("/leads", response_model=LeadOut)
def create_lead(payload: LeadIn) -> LeadOut:
    # Stub that confirms the API is live.
    # Replace with your real persistence later.
    return LeadOut(id="demo-lead-1", status="queued")


# ---------- Business (stub) ----------

@app.get("/business", response_model=BusinessOut)
async def get_business(
    business_name: str = Query(..., description="e.g., 'Il Lago Trattoria'"),
    location: str = Query(..., description="e.g., 'Doral, FL'"),
    target_category: Optional[str] = Query(None),
) -> BusinessOut:
    # Stub: just returns a shape compatible with your frontend.
    # Replace with your real Google Places logic later.
    return BusinessOut(
        found=True,
        name=business_name,
        formatted_address=f"{business_name}, {location}",
        rating=4.7,
        user_ratings_total=1234,
        website=None,
        categories=["restaurant", "italian_restaurant"],
        open_now=True,
        competitors=[
            {"place_id": "demo1", "name": "Nearby A", "rating": 4.6, "user_ratings_total": 987, "vicinity": location},
            {"place_id": "demo2", "name": "Nearby B", "rating": 4.5, "user_ratings_total": 543, "vicinity": location},
        ],
        error=None,
    )


# ---------- Score (stub) ----------

@app.get("/score", response_model=ScoreOut)
async def get_score(
    business_name: str = Query(...),
    location: str = Query(...),
    website: str = Query(...),
    target_category: Optional[str] = Query(None),
) -> ScoreOut:
    # Minimal deterministic scoring so the UI renders.
    perf = 0.50       # stub "raw"
    gbp_quality = 0.90
    comp_gap = 0.85

    # Default weights matching your earlier plan
    weights = {
        "performance": 0.30,
        "gbp_quality": 0.45,
        "category_match": 0.10,
        "competition_gap": 0.15,
    }

    components = [
        ScoreComponent(
            key="performance",
            weight=weights["performance"],
            raw=perf,
            points=round(weights["performance"] * perf * 100, 1),
            explain="PageSpeed (mobile PERFORMANCE category) [stub]",
        ),
        ScoreComponent(
            key="gbp_quality",
            weight=weights["gbp_quality"],
            raw=gbp_quality,
            points=round(weights["gbp_quality"] * gbp_quality * 100, 1),
            explain="GBP rating/reviews/presence/open_now [stub]",
        ),
        ScoreComponent(
            key="category_match",
            weight=weights["category_match"],
            raw=None if not target_category else 1.0,
            points=None if not target_category else round(weights["category_match"] * 100, 1),
            explain="GBP types vs. provided target_category [stub]",
        ),
        ScoreComponent(
            key="competition_gap",
            weight=weights["competition_gap"],
            raw=comp_gap,
            points=round(weights["competition_gap"] * comp_gap * 100, 1),
            explain="Your rating vs. nearby competitors’ avg [stub]",
        ),
    ]

    total = sum(c.points or 0 for c in components)
    score = int(round(total))

    return ScoreOut(
        input={
            "business_name": business_name,
            "location": location,
            "website": website,
            "target_category": target_category,
        },
        pagespeed={"performance_score": int(perf * 100), "error": None},
        gbp={"found": True, "name": business_name, "rating": 4.9, "user_ratings_total": 9999, "open_now": True, "categories": ["restaurant"]},
        components=components,
        score=score,
        warnings= None,
    )
