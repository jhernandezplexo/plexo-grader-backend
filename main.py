import os
import math
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

# -------------------------
# App
# -------------------------
app = FastAPI(title="Plexo Grader Backend", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
PAGESPEED_API_KEY = os.getenv("PAGESPEED_API_KEY", "").strip()


# -------------------------
# Helpers
# -------------------------
def _ok(v: Optional[str]) -> bool:
    return bool(v and v.strip())

def _clean_url(url: str) -> str:
    u = url.strip()
    if u and not u.startswith("http"):
        u = "https://" + u
    return u

def bayesian_rating(rating: float, reviews: int, m: int = 200, C: float = 4.3) -> float:
    """
    Bayesian smoothing to reduce the influence of small review counts.
    bayes = (m*C + n*R) / (m + n)
    """
    n = max(0, int(reviews))
    R = max(0.0, min(5.0, float(rating or 0)))
    return (m * C + n * R) / (m + n) if (m + n) > 0 else C

def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


# -------------------------
# Google APIs
# -------------------------
async def google_places_search(client: httpx.AsyncClient, business_name: str, location: str) -> Optional[Dict[str, Any]]:
    """
    Use Text Search to find a place. Returns first candidate or None.
    """
    if not _ok(GOOGLE_API_KEY):
        return None

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": f"{business_name} {location}", "key": GOOGLE_API_KEY}
    r = await client.get(url, params=params, timeout=30)
    data = r.json()
    candidates = data.get("results") or []
    return candidates[0] if candidates else None

async def google_place_details(client: httpx.AsyncClient, place_id: str) -> Optional[Dict[str, Any]]:
    if not _ok(GOOGLE_API_KEY):
        return None
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "key": GOOGLE_API_KEY,
        "fields": "place_id,name,formatted_address,types,website,rating,user_ratings_total,opening_hours,geometry,url"
    }
    r = await client.get(url, params=params, timeout=30)
    return (r.json() or {}).get("result")

async def google_nearby_by_type(
    client: httpx.AsyncClient,
    lat: float,
    lng: float,
    gtype: str,
    radius_m: int = 3000,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    if not _ok(GOOGLE_API_KEY):
        return []

    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius_m,
        "type": gtype,
        "key": GOOGLE_API_KEY,
    }
    r = await client.get(url, params=params, timeout=30)
    results = (r.json() or {}).get("results") or []
    out: List[Dict[str, Any]] = []
    for x in results[:max_results]:
        out.append({
            "place_id": x.get("place_id"),
            "name": x.get("name"),
            "rating": x.get("rating"),
            "user_ratings_total": x.get("user_ratings_total"),
            "vicinity": x.get("vicinity"),
            "types": x.get("types") or [],
        })
    return out

async def pagespeed_performance(client: httpx.AsyncClient, url: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Returns (0..1 score, error)
    """
    u = _clean_url(url)
    if not _ok(PAGESPEED_API_KEY):
        return None, "No PAGESPEED_API_KEY set"
    api = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
    params = {"url": u, "category": "PERFORMANCE", "strategy": "MOBILE", "key": PAGESPEED_API_KEY}
    r = await client.get(api, params=params, timeout=60)
    data = r.json()
    try:
        score = data["lighthouseResult"]["categories"]["performance"]["score"]
        return float(score), None
    except Exception:
        return None, "Unable to read PageSpeed score"


# -------------------------
# Schemas
# -------------------------
class ScoreComponent(BaseModel):
    key: str
    weight: float
    raw: Optional[float] = None
    explain: str
    points: Optional[float] = None

class ScoreResponse(BaseModel):
    input: Dict[str, Any]
    pagespeed: Dict[str, Any]
    gbp: Dict[str, Any]
    components: List[ScoreComponent]
    score: int
    warnings: Optional[str] = None


# -------------------------
# Routes
# -------------------------
@app.get("/", summary="Health")
async def health() -> Dict[str, str]:
    return {"ok": "true"}

@app.get("/business", summary="Business lookup")
async def business(
    business_name: str = Query(...),
    location: str = Query(...),
    target_category: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """
    Finds the business on Google, returns key GBP fields and a short competitor list
    (filtered by 'type' if a target_category is provided and maps to a Places type like 'italian_restaurant').
    """
    out: Dict[str, Any] = {"found": False, "error": None}
    async with httpx.AsyncClient() as client:
        candidate = await google_places_search(client, business_name, location)
        if not candidate:
            out["error"] = "Not found"
            return out

        out["found"] = True
        place_id = candidate["place_id"]
        detail = await google_place_details(client, place_id)
        if not detail:
            out["error"] = "Details not found"
            return out

        types = detail.get("types") or []
        opening = (detail.get("opening_hours") or {}).get("open_now")
        out.update({
            "place_id": place_id,
            "name": detail.get("name"),
            "formatted_address": detail.get("formatted_address"),
            "rating": detail.get("rating"),
            "user_ratings_total": detail.get("user_ratings_total"),
            "website": detail.get("website"),
            "google_maps_url": detail.get("url"),
            "categories": types,
            "open_now": bool(opening),
        })

        # Simple competitor sampler by type if the provided category looks like a Places type
        # e.g., "italian_restaurant", "pizza_restaurant", "coffee_shop", etc.
        competitors: List[Dict[str, Any]] = []
        if target_category and "_" in target_category:
            try:
                lat = detail["geometry"]["location"]["lat"]
                lng = detail["geometry"]["location"]["lng"]
                competitors = await google_nearby_by_type(client, lat, lng, target_category, radius_m=3500, max_results=12)
            except Exception:
                competitors = []
        out["competitors"] = competitors
        return out


@app.get("/score", response_model=ScoreResponse, summary="Score")
async def score(
    business_name: str = Query(..., example="Il Lago Trattoria"),
    location: str = Query(..., example="Doral"),
    website: str = Query(..., example="https://example.com/"),
    target_category: Optional[str] = Query(None, example="italian_restaurant"),
) -> ScoreResponse:
    """
    Returns an overall score (0–100) and a per-component breakdown.

    Weights (UI “Google visibility” mix):
      - performance (PageSpeed mobile):               30%
      - gbp_quality (rating/reviews/presence/open):   45%
      - category_match (if target_category given):    10%
      - competition_gap (your Bayesian rating vs avg):15%

    If a component cannot be computed, weights auto-rescale across the available components.
    """
    input_obj = {
        "business_name": business_name,
        "location": location,
        "website": website,
        "target_category": target_category,
    }

    async with httpx.AsyncClient() as client:
        # PageSpeed
        perf_raw, perf_err = await pagespeed_performance(client, website)
        pagespeed_obj = {"performance_score": round(perf_raw * 100) if perf_raw is not None else None, "error": perf_err}

        # GBP
        gbp_obj = {
            "found": False,
            "name": None,
            "rating": None,
            "user_ratings_total": None,
            "open_now": None,
            "categories": [],
        }
        biz = await business(business_name=business_name, location=location, target_category=target_category)
        if biz.get("found"):
            gbp_obj.update({
                "found": True,
                "name": biz.get("name"),
                "rating": biz.get("rating"),
                "user_ratings_total": biz.get("user_ratings_total"),
                "open_now": biz.get("open_now"),
                "categories": biz.get("categories") or [],
            })
        competitors = (biz.get("competitors") or []) if isinstance(biz, dict) else []

        # ---------- Components ----------
        components: List[ScoreComponent] = []
        desired_weights = {
            "performance": 0.30,
            "gbp_quality": 0.45,
            "category_match": 0.10,
            "competition_gap": 0.15,
        }

        available_keys: List[str] = []

        # performance
        if perf_raw is not None:
            components.append(ScoreComponent(
                key="performance",
                weight=desired_weights["performance"],
                raw=perf_raw,
                explain="PageSpeed (mobile PERFORMANCE category)"
            ))
            available_keys.append("performance")

        # gbp_quality
        if gbp_obj["found"]:
            # rating 0..5 -> 0..1
            rating_norm = clamp01((float(gbp_obj["rating"] or 0) / 5.0))
            # reviews: log scale to 0..1 (5000+ ~ 1.0)
            reviews = int(gbp_obj["user_ratings_total"] or 0)
            reviews_factor = clamp01(math.log10(max(1, reviews)) / math.log10(5000)) if reviews > 0 else 0.0
            # presence: if the profile exists, presence=1
            presence = 1.0
            # open now: small 0.05 bump when open
            open_now = 0.05 if bool(gbp_obj["open_now"]) else 0.0

            raw_gbp = clamp01(0.7 * rating_norm + 0.25 * reviews_factor + 0.05 * presence + open_now)
            components.append(ScoreComponent(
                key="gbp_quality",
                weight=desired_weights["gbp_quality"],
                raw=raw_gbp,
                explain="GBP rating/reviews/presence/open_now"
            ))
            available_keys.append("gbp_quality")

        # category_match
        if gbp_obj["found"] and target_category:
            types = set(gbp_obj.get("categories") or [])
            raw_cat = 1.0 if target_category in types else 0.0
            components.append(ScoreComponent(
                key="category_match",
                weight=desired_weights["category_match"],
                raw=raw_cat,
                explain="GBP types vs. provided target_category",
            ))
            available_keys.append("category_match")

        # competition_gap (Bayesian)
        if gbp_obj["found"] and competitors:
            my_bayes = bayesian_rating(gbp_obj["rating"] or 0.0, int(gbp_obj["user_ratings_total"] or 0))
            bayes_list: List[float] = []
            for c in competitors:
                r = float(c.get("rating") or 0.0)
                n = int(c.get("user_ratings_total") or 0)
                bayes_list.append(bayesian_rating(r, n))
            comp_avg = sum(bayes_list) / len(bayes_list) if bayes_list else my_bayes

            # map difference (-2..+2 roughly) into 0..1 with 0.5 at parity
            diff = my_bayes - comp_avg
            raw_gap = clamp01(0.5 + diff / 4.0)  # +4pts => +1.0, -4pts => 0.0
            components.append(ScoreComponent(
                key="competition_gap",
                weight=desired_weights["competition_gap"],
                raw=raw_gap,
                explain="Your Bayesian rating vs nearby competitors’ avg",
            ))
            available_keys.append("competition_gap")

        # Auto-rescale weights to only what's available
        total_w = sum(desired_weights[k] for k in available_keys) or 1.0
        for comp in components:
            comp.weight = desired_weights[comp.key] / total_w
            comp.points = round((comp.raw or 0.0) * comp.weight * 100, 1)

        total_points = sum((c.points or 0.0) for c in components)
        overall_score = int(round(total_points))

        return ScoreResponse(
            input=input_obj,
            pagespeed=pagespeed_obj,
            gbp=gbp_obj,
            components=components,
            score=overall_score,
            warnings=None
        )
