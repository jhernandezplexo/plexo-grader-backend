# main.py
import os
import math
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# ----------------------------
# FastAPI app + CORS (Render)
# ----------------------------
app = FastAPI(title="Plexo Grader API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten later (e.g., your Vercel domain)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Environment / constants
# ----------------------------
GMAPS_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
PSI_ENDPOINT = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
PLACES_TEXTSEARCH = "https://maps.googleapis.com/maps/api/place/textsearch/json"
PLACES_DETAILS = "https://maps.googleapis.com/maps/api/place/details/json"
PLACES_NEARBY = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

HTTP_TIMEOUT = 25.0

# Map friendly category names to Google Place types
CATEGORY_TO_TYPES = {
    "italian restaurant": ["italian_restaurant"],
    "restaurant": ["restaurant"],
    "pizza": ["pizza_restaurant", "restaurant"],
    "coffee": ["cafe", "coffee_shop"],
    "bakery": ["bakery"],
    "bar": ["bar"],
}


# ----------------------------
# Utilities
# ----------------------------
def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def bayesian_rating(
    R: float, n: int, C: float = 4.3, m: int = 200
) -> float:
    """Bayesian average to avoid 5.0/8 reviews outranking 4.7/3000."""
    if n < 0:
        n = 0
    return (m * C + n * R) / (m + n)


def rescale_0_1(value: float, min_v: float, max_v: float) -> float:
    if max_v <= min_v:
        return 0.0
    return max(0.0, min(1.0, (value - min_v) / (max_v - min_v)))


def best_category_types(
    explicit_target: Optional[str], place_types: List[str]
) -> List[str]:
    # prefer explicit target mapping
    if explicit_target:
        key = explicit_target.strip().lower()
        if key in CATEGORY_TO_TYPES:
            return CATEGORY_TO_TYPES[key]
        # fall back to a single normalized token (e.g., "italian")
        guess = f"{key.replace(' ', '_')}"
        return [guess]

    # otherwise, infer from the place details' types
    # pick the first restaurant-like type if present
    for t in place_types:
        if t.endswith("_restaurant") or t in {"restaurant", "cafe", "bar", "bakery"}:
            return [t]
    return ["restaurant"]


async def gmaps_get(client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    params = {**params, "key": GMAPS_KEY}
    r = await client.get(url, params=params, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


async def find_business(
    client: httpx.AsyncClient, business_name: str, location: str
) -> Optional[Dict[str, Any]]:
    """Use Text Search to find a business candidate by name + location."""
    if not GMAPS_KEY:
        return None
    q = f"{business_name} {location}".strip()
    data = await gmaps_get(client, PLACES_TEXTSEARCH, {"query": q})
    results = data.get("results", [])
    return results[0] if results else None


async def get_place_details(client: httpx.AsyncClient, place_id: str) -> Optional[Dict[str, Any]]:
    if not GMAPS_KEY:
        return None
    data = await gmaps_get(
        client,
        PLACES_DETAILS,
        {
            "place_id": place_id,
            "fields": "place_id,name,formatted_address,types,website,rating,user_ratings_total,opening_hours,geometry",
        },
    )
    return data.get("result")


async def nearby_competitors(
    client: httpx.AsyncClient,
    lat: float,
    lng: float,
    radius_m: int,
    types_filter: List[str],
) -> List[Dict[str, Any]]:
    """Find nearby places and filter by type(s)."""
    if not GMAPS_KEY:
        return []

    params = {
        "location": f"{lat},{lng}",
        "radius": radius_m,
        "type": "restaurant",  # broad seed (we’ll filter further)
    }

    data = await gmaps_get(client, PLACES_NEARBY, params)
    results = data.get("results", [])

    filtered: List[Dict[str, Any]] = []
    for r in results:
        r_types = [t for t in r.get("types", [])]
        # keep if ANY desired type appears in result types
        if any(t in r_types for t in types_filter):
            filtered.append(
                {
                    "place_id": r.get("place_id"),
                    "name": r.get("name"),
                    "rating": safe_float(r.get("rating")),
                    "user_ratings_total": int(r.get("user_ratings_total", 0)),
                    "vicinity": r.get("vicinity"),
                    "types": r_types,
                }
            )
    # sort by bayesian strength then rating
    def key_fn(x: Dict[str, Any]) -> Tuple[float, float]:
        b = bayesian_rating(x["rating"], x["user_ratings_total"])
        return (b, x["rating"])

    filtered.sort(key=key_fn, reverse=True)
    return filtered[:10]


async def pagespeed_performance(client: httpx.AsyncClient, url: str) -> Tuple[Optional[int], Optional[str]]:
    """Return mobile PERFORMANCE category score (0–100) using PSI v5."""
    try:
        resp = await client.get(
            PSI_ENDPOINT,
            params={
                "url": url,
                "strategy": "mobile",
                "category": "performance",
            },
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # PSI returns score in [0,1]; multiply to 0..100
        score = data.get("lighthouseResult", {}) \
                    .get("categories", {}) \
                    .get("performance", {}) \
                    .get("score", None)
        if score is None:
            return None, "No performance score in PSI response"
        return int(round(score * 100)), None
    except Exception as e:
        return None, str(e)


# ----------------------------
# Endpoints
# ----------------------------

@app.get("/scan")
async def scan(
    name: str = Query(...),
    email: str = Query(...),
    business_name: str = Query(""),
    website: str = Query(""),
):
    """Simple endpoint your / (home) page calls — returns PSI score only."""
    async with httpx.AsyncClient() as client:
        perf, err = (None, None)
        if website:
            perf, err = await pagespeed_performance(client, website)

    return {
        "name": name,
        "email": email,
        "business_name": business_name,
        "website": website,
        "performance_score": perf,
        "psi_error": err,
    }


@app.get("/business", summary="Business + competitors (category-filtered)")
async def business(
    business_name: str = Query(..., description="e.g., 'Il Lago Trattoria'"),
    location: str = Query(..., description="e.g., 'Doral, FL'"),
    target_category: Optional[str] = Query(None, description="e.g., 'Italian restaurant'"),
    radius_m: int = Query(2000, description="Search radius for competitors (meters)"),
):
    async with httpx.AsyncClient() as client:
        place = await find_business(client, business_name, location)
        if not place:
            return {"found": False, "error": "Business not found"}

        details = await get_place_details(client, place.get("place_id", ""))
        if not details:
            return {"found": False, "error": "Details not found"}

        # infer types for competitor filtering
        desired_types = best_category_types(target_category, details.get("types", []))

        geom = details.get("geometry", {}).get("location", {})
        lat, lng = geom.get("lat"), geom.get("lng")

        comps: List[Dict[str, Any]] = []
        if lat is not None and lng is not None:
            comps = await nearby_competitors(client, lat, lng, radius_m, desired_types)

        payload = {
            "found": True,
            "place_id": details.get("place_id"),
            "name": details.get("name"),
            "formatted_address": details.get("formatted_address"),
            "rating": safe_float(details.get("rating")),
            "user_ratings_total": int(details.get("user_ratings_total", 0)),
            "website": details.get("website"),
            "categories": details.get("types", []),
            "open_now": bool(details.get("opening_hours", {}).get("open_now")) if details.get("opening_hours") else None,
            "google_maps_url": f"https://maps.google.com/?q=place_id:{details.get('place_id')}",
            "competitors": comps,
            "error": None,
        }
        return payload


@app.get("/score", summary="Score")
async def score(
    business_name: str = Query(..., description="e.g., 'Il Lago Trattoria'"),
    location: str = Query(..., description="e.g., 'Doral'"),
    website: str = Query(..., description="https://example.com/"),
    target_category: Optional[str] = Query(None, description="Optional, e.g., 'Italian restaurant'"),
):
    """
    Returns overall score (0–100) and component breakdown.
    Weights (your 'Local-SEO heavier' spec):
      - performance (PageSpeed)............... 30%
      - gbp_quality (rating/reviews/presence).. 45%
      - category_match (if provided)........... 10%
      - competition_gap (vs nearby avg)........ 15%
    Missing components auto-rescale.
    """
    weights = {
        "performance": 0.30,
        "gbp_quality": 0.45,
        "category_match": 0.10,
        "competition_gap": 0.15,
    }

    async with httpx.AsyncClient() as client:
        # 1) PageSpeed
        perf_score, perf_err = await pagespeed_performance(client, website)
        perf_raw = None if perf_score is None else max(0.0, min(1.0, perf_score / 100.0))

        # 2) GBP details + competitors
        place = await find_business(client, business_name, location)
        if not place:
            # If we can’t find GBP, still return what we have (performance)
            active = []
            if perf_raw is not None:
                active.append(("performance", perf_raw, "PageSpeed (mobile PERFORMANCE category)"))
            total_w = sum(weights[k] for k, _, _ in active)
            components = []
            total_points = 0.0
            for k, raw, explain in active:
                w = weights[k] / total_w if total_w else 0
                pts = raw * (w * 100)
                total_points += pts
                components.append({"key": k, "weight": w, "raw": raw, "explain": explain, "points": round(pts, 1)})
            return {
                "input": {"business_name": business_name, "location": location, "website": website, "target_category": target_category},
                "pagespeed": {"performance_score": perf_score, "error": perf_err},
                "gbp": {"found": False},
                "components": components,
                "score": int(round(total_points)),
                "warnings": "GBP not found",
            }

        details = await get_place_details(client, place.get("place_id", ""))
        details = details or {}
        name = details.get("name")
        rating = safe_float(details.get("rating"))
        reviews = int(details.get("user_ratings_total", 0))
        website_gbp = details.get("website")
        types = details.get("types", [])
        open_now = bool(details.get("opening_hours", {}).get("open_now")) if details.get("opening_hours") else None

        # category selection for comps
        desired_types = best_category_types(target_category, types)
        geom = details.get("geometry", {}).get("location", {})
        lat, lng = geom.get("lat"), geom.get("lng")

        comps: List[Dict[str, Any]] = []
        if lat is not None and lng is not None:
            comps = await nearby_competitors(client, lat, lng, 2000, desired_types)

        # 2a) GBP quality raw (0..1): combine (bayesian rating normalized) + presence bonus + open_now bonus
        b_rating = bayesian_rating(rating, reviews) if rating else 0.0
        # normalize ratings around ~ [3.5..5.0] -> [0..1]
        rating_norm = rescale_0_1(b_rating, 3.5, 5.0)
        presence_bonus = 0.05 if website_gbp else 0.0
        open_bonus = 0.02 if open_now else 0.0
        gbp_raw = max(0.0, min(1.0, rating_norm + presence_bonus + open_bonus))

        # 2b) category match raw (0..1): if target provided, 1 if any desired type overlaps GBP types; else None
        if target_category:
            match = 1.0 if any(t in types for t in desired_types) else 0.0
            category_raw: Optional[float] = match
        else:
            category_raw = None

        # 2c) competition gap raw (0..1): your bayesian vs comps bayesian avg
        comp_avg = None
        if comps:
            comp_bayes = [bayesian_rating(safe_float(c["rating"]), int(c["user_ratings_total"])) for c in comps if c.get("rating") is not None]
            if comp_bayes:
                comp_avg = sum(comp_bayes) / len(comp_bayes)

        gap_raw = None
        if comp_avg is not None and rating:
            # advantage = your_bayes - competitors_avg; map [-1.0 .. +1.0] roughly into [0..1]
            adv = bayesian_rating(rating, reviews) - comp_avg
            gap_raw = max(0.0, min(1.0, (adv + 1.0) / 2.0))

        # collect active components & rescale weights to available signals
        active: List[Tuple[str, Optional[float], str]] = [
            ("performance", perf_raw, "PageSpeed (mobile PERFORMANCE category)"),
            ("gbp_quality", gbp_raw, "GBP rating/reviews/presence/open_now"),
            ("category_match", category_raw, "GBP types vs. provided target_category"),
            ("competition_gap", gap_raw, "Your rating vs. nearby competitors’ avg"),
        ]
        available = [(k, r, e) for (k, r, e) in active if r is not None]
        total_w = sum(weights[k] for k, _, _ in available)

        components = []
        total_points = 0.0
        for k, raw, explain in available:
            w = weights[k] / total_w if total_w else 0.0
            pts = raw * (w * 100.0)
            total_points += pts
            components.append(
                {
                    "key": k,
                    "weight": round(w, 2),
                    "raw": round(raw, 3),
                    "explain": explain,
                    "points": round(pts, 1),
                }
            )

        return {
            "input": {
                "business_name": business_name,
                "location": location,
                "website": website,
                "target_category": target_category,
            },
            "pagespeed": {"performance_score": perf_score, "error": perf_err},
            "gbp": {
                "found": True,
                "name": name,
                "rating": rating,
                "user_ratings_total": reviews,
                "open_now": open_now,
                "categories": types,
                "website": website_gbp,
            },
            "competitors_summary": {
                "desired_types": desired_types,
                "count": len(comps),
                "examples": comps[:5],  # small peek
            },
            "components": components,
            "score": int(round(total_points)),
            "warnings": None,
        }


# ---------------
# Root (optional)
# ---------------
@app.get("/")
def root():
    return {"ok": True, "service": "Plexo Grader API"}
