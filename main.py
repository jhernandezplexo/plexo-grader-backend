# main.py
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# ---------------------------
# App + CORS (open for now)
# ---------------------------
app = FastAPI(title="Plexo Grader Backend", version="0.3.0")

# ---------------------------
# Env
# ---------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
PAGESPEED_API_KEY = os.getenv("PAGESPEED_API_KEY", "").strip()

# ---------------------------
# Helpers
# ---------------------------

def _ok(v: Optional[str]) -> bool:
    return bool(v and v.strip())

def _clean_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        return u
    if u.startswith("http://") or u.startswith("https://"):
        return u
    # default to https
    return "https://" + u

def bayesian_rating(rating: float, reviews: int, m: int = 200, C: float = 4.3) -> float:
    """
    Bayesian smoothing so 5.0/8 reviews < 4.7/3000 reviews.
    (m, C) are the prior review count and prior rating.
    """
    n = max(0, reviews)
    return (m * C + n * rating) / (m + n)

def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

# ---------------------------
# Models
# ---------------------------

class Competitor(BaseModel):
    place_id: str
    name: str
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    vicinity: Optional[str] = None

class BusinessOut(BaseModel):
    found: bool = False
    place_id: Optional[str] = None
    name: Optional[str] = None
    formatted_address: Optional[str] = None
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    open_now: Optional[bool] = None
    categories: List[str] = Field(default_factory=list)
    website: Optional[str] = None
    google_maps_url: Optional[str] = None
    competitors: List[Competitor] = Field(default_factory=list)
    error: Optional[str] = None

class ScoreComponent(BaseModel):
    key: str
    weight: float
    raw: Optional[float] = None
    explain: str
    points: Optional[float] = None

class ScoreOut(BaseModel):
    input: Dict[str, Any]
    pagespeed: Dict[str, Any]
    gbp: Dict[str, Any]
    components: List[ScoreComponent]
    score: int
    warnings: Optional[str] = None

# ---------------------------
# Google APIs (httpx on-demand)
# ---------------------------

PLACES_TEXT_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
PLACES_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
PAGESPEED_URL = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"

async def places_text_search(client: httpx.AsyncClient, query: str) -> Optional[Dict[str, Any]]:
    if not _ok(GOOGLE_API_KEY):
        return None
    params = {"query": query, "key": GOOGLE_API_KEY}
    r = await client.get(PLACES_TEXT_URL, params=params, timeout=30)
    data = r.json()
    results = data.get("results") or []
    return results[0] if results else None

async def places_details(client: httpx.AsyncClient, place_id: str) -> Optional[Dict[str, Any]]:
    if not _ok(GOOGLE_API_KEY):
        return None
    params = {
        "place_id": place_id,
        "fields": "place_id,name,formatted_address,types,rating,user_ratings_total,opening_hours,website,url,geometry",
        "key": GOOGLE_API_KEY,
    }
    r = await client.get(PLACES_DETAILS_URL, params=params, timeout=30)
    return r.json().get("result")

async def places_nearby_by_type(
    client: httpx.AsyncClient,
    lat: float,
    lng: float,
    gtype: str,
    radius_m: int = 3000,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    if not _ok(GOOGLE_API_KEY):
        return []
    params = {
        "location": f"{lat},{lng}",
        "radius": radius_m,
        "type": gtype,
        "key": GOOGLE_API_KEY,
    }
    r = await client.get(PLACES_NEARBY_URL, params=params, timeout=30)
    results = r.json().get("results") or []
    out: List[Dict[str, Any]] = []
    for x in results[:max_results]:
        out.append(
            {
                "place_id": x.get("place_id"),
                "name": x.get("name"),
                "rating": x.get("rating"),
                "user_ratings_total": x.get("user_ratings_total"),
                "vicinity": x.get("vicinity"),
                "types": x.get("types") or [],
            }
        )
    return out

async def run_pagespeed(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    if not _ok(PAGESPEED_API_KEY):
        return {"performance_score": None, "error": "missing PAGESPEED_API_KEY"}
    params = {"url": url, "strategy": "mobile", "key": PAGESPEED_API_KEY, "category": "PERFORMANCE"}
    r = await client.get(PAGESPEED_URL, params=params, timeout=60)
    j = r.json()
    score = j.get("lighthouseResult", {}).get("categories", {}).get("performance", {}).get("score")
    perf = None if score is None else float(score)
    return {"performance_score": None if perf is None else round(perf * 100), "raw": perf, "error": None}

# ---------------------------
# Endpoint: /health
# ---------------------------

@app.get("/health")
def health():
    return {"ok": True}

# ---------------------------
# Endpoint: /business
# ---------------------------

@app.get("/business", response_model=BusinessOut, summary="Find a business + competitors")
async def business(
    business_name: str = Query(..., description="e.g., 'Il Lago Trattoria'"),
    location: str = Query(..., description="e.g., 'Doral, FL'"),
    target_category: Optional[str] = Query(None, description="e.g., 'italian restaurant'")
):
    """
    1) Text search to find the business
    2) Details
    3) Nearby competitors (same category if provided)
    """
    out = BusinessOut()

    async with httpx.AsyncClient() as client:
        # 1) search
        q = f"{business_name} {location}".strip()
        candidate = await places_text_search(client, q)
        if not candidate:
            out.error = "Not found (text search)"
            return out

        place_id = candidate.get("place_id")
        # 2) details
        details = await places_details(client, place_id) or {}
        out.found = True
        out.place_id = details.get("place_id")
        out.name = details.get("name")
        out.formatted_address = details.get("formatted_address")
        out.rating = details.get("rating")
        out.user_ratings_total = details.get("user_ratings_total")
        out.open_now = (details.get("opening_hours") or {}).get("open_now")
        out.categories = details.get("types") or []
        out.website = details.get("website")
        out.google_maps_url = details.get("url")

        # 3) competitors
        comps: List[Competitor] = []
        loc = (details.get("geometry") or {}).get("location") or {}
        lat, lng = loc.get("lat"), loc.get("lng")

        # decide the 'type' to use for nearby search
        gtype = None
        if target_category:
            # crude mapping; Google type list prefers underscores
            gtype = target_category.lower().replace(" ", "_")
        else:
            # pick first place type that ends with '_restaurant' or is 'restaurant'
            for t in out.categories:
                if t == "restaurant" or t.endswith("_restaurant"):
                    gtype = t
                    break
        if lat is not None and lng is not None and gtype:
            raw = await places_nearby_by_type(client, lat, lng, gtype, radius_m=3000, max_results=15)
            for x in raw:
                comps.append(
                    Competitor(
                        place_id=x.get("place_id", ""),
                        name=x.get("name", ""),
                        rating=x.get("rating"),
                        user_ratings_total=x.get("user_ratings_total"),
                        vicinity=x.get("vicinity"),
                    )
                )
        out.competitors = comps[:10]
        return out

# ---------------------------
# Endpoint: /score
# ---------------------------

@app.get("/score", response_model=ScoreOut, summary="Compute overall score (0–100) and breakdown")
async def score(
    business_name: str = Query(...),
    location: str = Query(...),
    website: str = Query(...),
    target_category: Optional[str] = Query(None),
):
    weights = {
        "performance": 0.30,
        "gbp_quality": 0.45,
        "category_match": 0.10,
        "competition_gap": 0.15,
    }

    website_clean = _clean_url(website)

    async with httpx.AsyncClient() as client:
        # GBP lookup (reuse /business logic quickly)
        b = await business(business_name=business_name, location=location, target_category=target_category)
        # unwrap model (already BusinessOut)
        gbp_dict = b.model_dump() if isinstance(b, BusinessOut) else b

        # PageSpeed
        ps = await run_pagespeed(client, website_clean)
        perf_raw = ps.get("raw")  # 0..1 or None
        perf_points = weights["performance"] * 100.0 * (perf_raw if perf_raw is not None else 0.0)

        # GBP quality proxy: normalize 0..1 using rating and presence/open_now
        rating = gbp_dict.get("rating") or 0.0
        reviews = gbp_dict.get("user_ratings_total") or 0
        open_now = bool(gbp_dict.get("open_now"))
        presence_bonus = 0.05 if _ok(gbp_dict.get("website")) else 0.0
        # map rating (0..5) → 0..1
        rating_norm = clamp01((rating - 3.0) / 2.0)  # ~3.0 → 0, 5.0 → 1
        gbp_raw = clamp01(0.85 * rating_norm + 0.10 * (1.0 if open_now else 0.0) + presence_bonus)
        gbp_points = weights["gbp_quality"] * 100.0 * gbp_raw

        # Category match
        cat_raw = None
        if target_category:
            t = target_category.lower().replace(" ", "_")
            types = gbp_dict.get("categories") or []
            cat_raw = 1.0 if t in types else 0.0
        cat_points = None if cat_raw is None else weights["category_match"] * 100.0 * cat_raw

        # Competition gap – compare business bayesian rating vs avg of competitors
        comps = gbp_dict.get("competitors") or []
        my_bayes = bayesian_rating(rating or 0.0, reviews or 0)
        if comps:
            comp_bayes_vals = [
                bayesian_rating(float(c.get("rating") or 0.0), int(c.get("user_ratings_total") or 0))
                for c in comps
            ]
            avg_comp = sum(comp_bayes_vals) / max(1, len(comp_bayes_vals))
        else:
            avg_comp = my_bayes
        gap = clamp01(0.5 + 0.5 * (my_bayes - avg_comp))  # center @ equal
        gap_points = weights["competition_gap"] * 100.0 * gap

        components: List[ScoreComponent] = [
            ScoreComponent(key="performance", weight=weights["performance"], raw=perf_raw, explain="PageSpeed (mobile PERFORMANCE category)", points=round(perf_points, 1)),
            ScoreComponent(key="gbp_quality", weight=weights["gbp_quality"], raw=round(gbp_raw, 3), explain="GBP rating/presence/open_now", points=round(gbp_points, 1)),
            ScoreComponent(key="category_match", weight=weights["category_match"], raw=None if cat_raw is None else round(cat_raw, 3), explain="GBP types vs. provided target_category", points=None if cat_points is None else round(cat_points, 1)),
            ScoreComponent(key="competition_gap", weight=weights["competition_gap"], raw=round(gap, 3), explain="Your bayesian rating vs. nearby competitors’ avg", points=round(gap_points, 1)),
        ]

        total_points = sum([c.points or 0.0 for c in components])
        final_score = int(round(total_points))

        return ScoreOut(
            input={"business_name": business_name, "location": location, "website": website_clean, "target_category": target_category},
            pagespeed={"performance_score": ps.get("performance_score"), "error": ps.get("error")},
            gbp=gbp_dict,
            components=components,
            score=final_score,
            warnings=None if (_ok(GOOGLE_API_KEY) and _ok(PAGESPEED_API_KEY)) else "Missing one or more API keys; some components degraded.",
        )

# --------------- Run local (optional) ---------------
# On Render this block is ignored (they run uvicorn via start command)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
