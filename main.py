# main.py
import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

app = FastAPI(
    title="Plexo Grader Backend",
    version="0.3.0",
)

# ---- Config / env -----------------------------------------------------------
GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or "").strip()
PAGESPEED_API_KEY = (os.getenv("PAGESPEED_API_KEY") or "").strip()

PLACES_TEXT_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
PLACES_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
PAGESPEED_URL = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"


# ---- Models -----------------------------------------------------------------
class Competitor(BaseModel):
    place_id: str
    name: str
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    vicinity: Optional[str] = None
    types: List[str] = Field(default_factory=list)


class BusinessOut(BaseModel):
    found: bool = False
    place_id: Optional[str] = None

    name: Optional[str] = None
    formatted_address: Optional[str] = None
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    website: Optional[str] = None
    google_maps_url: Optional[str] = None
    open_now: Optional[bool] = None
    categories: List[str] = Field(default_factory=list)

    competitors: List[Competitor] = Field(default_factory=list)
    error: Optional[str] = None


class ScoreComponent(BaseModel):
    key: str
    weight: float
    raw: float
    points: float
    explain: str


class ScoreOut(BaseModel):
    score: int
    components: List[ScoreComponent]
    website: Optional[str] = None
    place_id: Optional[str] = None
    name: Optional[str] = None
    error: Optional[str] = None


# ---- Helpers ----------------------------------------------------------------
def _need_google() -> None:
    if not GOOGLE_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Missing GOOGLE_API_KEY (backend not configured).",
        )


def _clean_url(u: Optional[str]) -> Optional[str]:
    if not u:
        return None
    u = u.strip()
    if not u:
        return None
    if u.startswith("http://") or u.startswith("https://"):
        return u
    return "https://" + u


async def places_text_search(client: httpx.AsyncClient, query: str) -> Optional[Dict[str, Any]]:
    params = {"query": query, "key": GOOGLE_API_KEY}
    r = await client.get(PLACES_TEXT_URL, params=params, timeout=30)
    data = r.json() or {}
    results = data.get("results") or []
    return results[0] if results else None


async def places_details(
    client: httpx.AsyncClient,
    place_id: str,
) -> Optional[Dict[str, Any]]:
    params = {
        "place_id": place_id,
        "key": GOOGLE_API_KEY,
        "fields": "place_id,name,formatted_address,types,rating,user_ratings_total,opening_hours,website,url,geometry",
    }
    r = await client.get(PLACES_DETAILS_URL, params=params, timeout=30)
    data = r.json() or {}
    return data.get("result")


async def places_nearby_by_type(
    client: httpx.AsyncClient,
    lat: float,
    lng: float,
    gtype: str,
    radius_m: int = 3000,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    params = {
        "location": f"{lat},{lng}",
        "radius": radius_m,
        "type": gtype,
        "key": GOOGLE_API_KEY,
    }
    r = await client.get(PLACES_NEARBY_URL, params=params, timeout=30)
    results = (r.json() or {}).get("results") or []
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


async def pagespeed_mobile_score(client: httpx.AsyncClient, website: Optional[str]) -> Optional[float]:
    if not PAGESPEED_API_KEY or not website:
        return None
    params = {
        "key": PAGESPEED_API_KEY,
        "url": website,
        "strategy": "MOBILE",
        "category": "PERFORMANCE",
    }
    r = await client.get(PAGESPEED_URL, params=params, timeout=45)
    data = r.json() or {}
    try:
        cat = data["lighthouseResult"]["categories"]["performance"]["score"]
        # Lighthouse score comes 0..1
        return float(cat) if isinstance(cat, (float, int)) else None
    except Exception:
        return None


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def bayesian_rating(rating: float, reviews: int, m: int = 200, C: float = 4.3) -> float:
    """Bayesian smoothing for ratings with low review counts."""
    n = float(reviews or 0)
    return (m * C + n * rating) / (m + n)


# ---- Routes -----------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"ok": "true"}


@app.get(
    "/business",
    response_model=BusinessOut,
    summary="Business lookup (GBP + competitors)",
)
async def business(
    business: str = Query(..., description="Business name (e.g. 'Il Lago Trattoria')"),
    location: str = Query(..., description="City/Area (e.g. 'Doral')"),
) -> BusinessOut:
    """
    Finds a place via Text Search, then fetches Details and nearby competitors
    using the primary type from the Details result.
    """
    _need_google()

    query = f"{business} {location}".strip()
    out = BusinessOut()

    async with httpx.AsyncClient() as client:
        candidate = await places_text_search(client, query)
        if not candidate:
            return BusinessOut(found=False, error="No match from Text Search.")

        place_id = candidate.get("place_id")
        details = await places_details(client, place_id)
        if not details:
            return BusinessOut(found=False, error="No Details for candidate.")

        # Fill details
        out.found = True
        out.place_id = place_id
        out.name = details.get("name")
        out.formatted_address = details.get("formatted_address")
        out.rating = details.get("rating")
        out.user_ratings_total = details.get("user_ratings_total")
        out.website = _clean_url(details.get("website"))
        out.google_maps_url = details.get("url")
        out.open_now = (details.get("opening_hours") or {}).get("open_now")
        out.categories = details.get("types") or []

        # Competitors by primary type (if any) within 3km
        geom = details.get("geometry") or {}
        loc = (geom.get("location") or {})
        lat, lng = loc.get("lat"), loc.get("lng")
        primary_type = out.categories[0] if out.categories else None

        if lat and lng and primary_type:
            comps = await places_nearby_by_type(
                client, lat=lat, lng=lng, gtype=primary_type, radius_m=3000, max_results=10
            )
            out.competitors = [Competitor(**c) for c in comps]

    return out


@app.get(
    "/score",
    response_model=ScoreOut,
    summary="Score",
)
async def score(
    name: str = Query(..., description="Business name"),
    location: str = Query(..., description="City/Area"),
    website: Optional[str] = Query(None, description="Website URL (optional)"),
    target_category: Optional[str] = Query(None, description="Target GBP category (optional)"),
) -> ScoreOut:
    """
    Simple scoring made of:
      - performance (PageSpeed mobile) 40%
      - gbp_quality (rating + presence + open_now) 35%
      - category_match 10% (if target_category provided)
      - competition_gap 15% (your rating vs avg of nearby competitors)
    """
    _need_google()
    site = _clean_url(website)

    async with httpx.AsyncClient() as client:
        # 1) Find place + details (and competitors)
        b = await business(business=name, location=location)
        if not b.found:
            return ScoreOut(score=0, components=[], website=site, error=b.error or "Not found")

        # Ensure we have a final website (prefer details.website if any)
        site = _clean_url(b.website) or site

        # 2) PageSpeed performance
        perf_raw = await pagespeed_mobile_score(client, site)
        perf_points = 0.0 if perf_raw is None else 100.0 * perf_raw  # 0..100 from 0..1
        comp_performance = ScoreComponent(
            key="performance",
            weight=0.40,
            raw=0.0 if perf_raw is None else float(perf_raw),
            points=perf_points * 0.40 / 100.0,
            explain="PageSpeed (mobile PERFORMANCE category)",
        )

        # 3) GBP quality (rating normalized + presence + open_now)
        rating = float(b.rating or 0.0)
        reviews = int(b.user_ratings_total or 0)
        bayes = bayesian_rating(rating, reviews)  # ~0..5
        gbp_quality_raw = clamp01(bayes / 5.0)
        presence_bonus = 0.05 if b.google_maps_url else 0.0
        open_bonus = 0.05 if (b.open_now is True) else 0.0
        gbp_quality_score = clamp01(gbp_quality_raw + presence_bonus + open_bonus)
        comp_gbp = ScoreComponent(
            key="gbp_quality",
            weight=0.35,
            raw=float(gbp_quality_score),
            points=gbp_quality_score * 0.35 * 100.0 / 1.0 / 100.0,  # keep in 0..weight
            explain="GBP rating/reviews/presence/open_now",
        )

        # 4) Category match (optional)
        cat_points = 0.0
        cat_raw = 0.0
        if target_category:
            has_match = any(target_category.lower() == t.lower() for t in b.categories)
            cat_raw = 1.0 if has_match else 0.0
            cat_points = 0.10 if has_match else 0.0
        comp_cat = ScoreComponent(
            key="category_match",
            weight=0.10,
            raw=float(cat_raw),
            points=float(cat_points),
            explain="GBP types vs. provided target_category",
        )

        # 5) Competition gap (your rating vs avg competitors’ rating)
        your = float(b.rating or 0.0)
        opp_ratings = [float(c.rating or 0.0) for c in b.competitors if c.rating is not None]
        avg_opp = sum(opp_ratings) / len(opp_ratings) if opp_ratings else 0.0
        # map difference [-2..+2] -> [0..1] roughly
        diff = your - avg_opp
        comp_raw = clamp01((diff + 2.0) / 4.0)
        comp_gap = ScoreComponent(
            key="competition_gap",
            weight=0.15,
            raw=float(comp_raw),
            points=comp_raw * 0.15,
            explain="Your rating vs. nearby competitors’ avg",
        )

        # 6) Sum
        total_points = comp_performance.points + comp_gbp.points + comp_cat.points + comp_gap.points
        final_score = int(round(total_points * 100))  # 0..100

        return ScoreOut(
            score=final_score,
            components=[comp_performance, comp_gbp, comp_cat, comp_gap],
            website=site,
            place_id=b.place_id,
            name=b.name,
        )
