import os
import math
import json
import requests
from typing import List, Optional, Dict, Any, Tuple
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# -------------------------
# Config / Environment
# -------------------------
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
PAGESPEED_API_KEY = os.getenv("PAGESPEED_API_KEY", "")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")  # allow all if not set

# Safety checks
if not GOOGLE_PLACES_API_KEY:
    print("WARNING: GOOGLE_PLACES_API_KEY is not set. /business and /score will fail.")

# -------------------------
# FastAPI app (single!)
# -------------------------
app = FastAPI(title="Plexo Grader Backend", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Helpers: Google Places
# -------------------------

PLACES_TEXTSEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
PLACES_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

def _http_get(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def find_place(business_name: str, location: str) -> Optional[Dict[str, Any]]:
    """Use Text Search to find the place and return first candidate."""
    q = f"{business_name} {location}"
    data = _http_get(PLACES_TEXTSEARCH_URL, {
        "query": q,
        "key": GOOGLE_PLACES_API_KEY,
    })
    if data.get("status") != "OK" or not data.get("results"):
        return None
    return data["results"][0]

def get_place_details(place_id: str) -> Optional[Dict[str, Any]]:
    fields = ",".join([
        "place_id", "name", "formatted_address", "formatted_phone_number",
        "website", "url", "rating", "user_ratings_total",
        "opening_hours", "types", "geometry/location"
    ])
    data = _http_get(PLACES_DETAILS_URL, {
        "place_id": place_id,
        "fields": fields,
        "key": GOOGLE_PLACES_API_KEY,
    })
    if data.get("status") != "OK":
        return None
    return data.get("result", None)

def extract_primary_types(types: List[str]) -> List[str]:
    """Pick cuisine/restaurant-y types; prefer '*_restaurant' and 'restaurant'."""
    if not types:
        return []
    cuisine_types = [t for t in types if t.endswith("_restaurant")]
    if cuisine_types:
        return cuisine_types
    if "restaurant" in types:
        return ["restaurant"]
    # fallback to all types (rare)
    return types[:]

def nearby_competitors(lat: float, lng: float, radius_m: int = 1500) -> List[Dict[str, Any]]:
    """Find nearby places typed as restaurants within radius."""
    data = _http_get(PLACES_NEARBY_URL, {
        "location": f"{lat},{lng}",
        "radius": radius_m,
        "type": "restaurant",
        "key": GOOGLE_PLACES_API_KEY,
    })
    if data.get("status") != "OK":
        return []
    return data.get("results", [])

def filter_competitors_by_types(competitors: List[Dict[str, Any]],
                                target_types: List[str]) -> List[Dict[str, Any]]:
    """Return only comps sharing any of the target types."""
    if not target_types:
        return competitors
    target_set = set(target_types)
    filtered = []
    for c in competitors:
        c_types = set(c.get("types", []))
        if c_types & target_set:
            filtered.append(c)
    return filtered

def competitor_summary(comp: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "place_id": comp.get("place_id"),
        "name": comp.get("name"),
        "rating": comp.get("rating"),
        "user_ratings_total": comp.get("user_ratings_total"),
        "vicinity": comp.get("vicinity"),
        "types": comp.get("types"),
    }

# -------------------------
# Helper: PageSpeed Insights (mobile)
# -------------------------

PAGESPEED_URL = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"

def fetch_pagespeed_performance(url: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Returns (performance_score_0to100, error_message_or_None)
    Uses mobile (strategy='mobile').
    """
    try:
        params = {
            "url": url,
            "strategy": "mobile",
        }
        if PAGESPEED_API_KEY:
            params["key"] = PAGESPEED_API_KEY

        data = _http_get(PAGESPEED_URL, params)
        cat = data.get("lighthouseResult", {}).get("categories", {}).get("performance", {})
        score = cat.get("score", None)
        if score is None:
            return None, "No performance score in PSI response"
        # PSI returns 0..1; convert to 0..100
        return float(score) * 100.0, None
    except Exception as e:
        return None, str(e)

# -------------------------
# Helper: Bayesian smoothing
# -------------------------

def bayesian_rating(rating: Optional[float], n_reviews: Optional[int],
                    m_prior: int = 200, C_prior: float = 4.3) -> Optional[float]:
    """(m*C + n*R)/(m+n) — smoothed rating; returns None if rating missing."""
    if rating is None or n_reviews is None:
        return None
    try:
        n = max(0, int(n_reviews))
        smoothed = (m_prior * C_prior + n * float(rating)) / (m_prior + n)
        return smoothed
    except Exception:
        return None

def safe_div(a: float, b: float) -> float:
    return a / b if b not in (0, None) else 0.0

# -------------------------
# /business
# -------------------------

@app.get("/business", summary="Business lookup (GBP + competitors)")
def get_business(
    business_name: str = Query(..., description="Business name, e.g. 'Il Lago Trattoria'"),
    location: str = Query(..., description="City/State/Country, e.g. 'Doral, FL'")
):
    try:
        seed = find_place(business_name, location)
        if not seed:
            return {
                "found": False,
                "place_id": None,
                "name": None,
                "formatted_address": None,
                "rating": None,
                "user_ratings_total": None,
                "website": None,
                "google_maps_url": None,
                "categories": None,
                "open_now": None,
                "competitors": [],
                "error": "REQUEST_DENIED" if not GOOGLE_PLACES_API_KEY else "NOT_FOUND",
            }

        place_id = seed["place_id"]
        details = get_place_details(place_id)
        if not details:
            return {"found": False, "error": "DETAILS_NOT_FOUND"}

        name = details.get("name")
        addr = details.get("formatted_address")
        rating = details.get("rating")
        n_reviews = details.get("user_ratings_total")
        website = details.get("website")
        gmaps_url = details.get("url")
        types = details.get("types", [])
        opening_hours = details.get("opening_hours") or {}
        open_now = opening_hours.get("open_now", None)

        # competitors
        loc = details.get("geometry", {}).get("location", {})
        lat, lng = loc.get("lat"), loc.get("lng")
        comps = []
        if lat is not None and lng is not None:
            raw_comps = nearby_competitors(lat, lng, radius_m=1500)
            target_types = extract_primary_types(types)
            filtered = filter_competitors_by_types(raw_comps, target_types)
            # keep top by reviews, then rating
            filtered.sort(key=lambda x: (x.get("user_ratings_total", 0), x.get("rating", 0.0)), reverse=True)
            comps = [competitor_summary(c) for c in filtered[:10]]

        return {
            "found": True,
            "place_id": place_id,
            "name": name,
            "formatted_address": addr,
            "rating": rating,
            "user_ratings_total": n_reviews,
            "website": website,
            "google_maps_url": gmaps_url,
            "categories": types,
            "open_now": open_now,
            "competitors": comps,
            "error": None,
        }
    except Exception as e:
        return {
            "found": False,
            "error": str(e),
        }

# -------------------------
# /score
# -------------------------

@app.get("/score", summary="Score")
def score(
    business_name: str = Query(..., description="e.g. 'Il Lago Trattoria'"),
    location: str = Query(..., description="e.g. 'Doral' or 'Doral, FL'"),
    website: str = Query(..., description="https://example.com/"),
    target_category: Optional[str] = Query(None, description="Optional, e.g. 'italian restaurant'")
):
    """
    Returns an overall score (0–100) and a per-component breakdown.
    Weights (Local-SEO heavy): performance 30%, gbp_quality 45%, category_match 10%, competition_gap 15%.
    """
    # 1) PageSpeed
    psi_score, psi_err = fetch_pagespeed_performance(website)  # 0..100
    perf_norm = (psi_score or 0.0) / 100.0

    # 2) GBP data (+ comps)
    biz = get_business(business_name, location)
    gbp_err = biz.get("error")
    rating = biz.get("rating")
    n_reviews = biz.get("user_ratings_total")
    open_now = biz.get("open_now")
    categories = biz.get("categories") or []
    website_present = 1.0 if (biz.get("website") or "").strip() else 0.0

    # GBP quality: combine (smoothed rating / 5), review heft, presence, and open_now
    # rating_norm: use Bayesian smoothed rating / 5.0
    smoothed = bayesian_rating(rating, n_reviews, m_prior=200, C_prior=4.3)
    rating_norm = (smoothed or 0.0) / 5.0

    # presence boost: website present (0.05), open_now if available (0.05)
    presence = 0.0
    presence += 0.05 * website_present
    if open_now is not None:
        presence += 0.05 * (1.0 if open_now else 0.0)

    # review heft: log(1+reviews) normalized roughly vs 5000
    review_norm = min(1.0, math.log1p(n_reviews or 0) / math.log1p(5000))

    gbp_quality_raw = max(0.0, min(1.0, 0.85 * rating_norm + 0.10 * review_norm + presence))

    # 3) Category match: if target_category provided, measure overlap with types
    if target_category:
        tokens = [t.strip().lower() for t in target_category.split()]
        # any token appears in types string?
        types_str = " ".join(categories).lower()
        cat_match_raw = 1.0 if all(tok in types_str for tok in tokens if tok not in {"the", "and", "&"}) else 0.0
    else:
        cat_match_raw = None  # rescale weights later

    # 4) Competition gap: your smoothed rating vs competitors' smoothed mean
    comps = biz.get("competitors", [])
    comp_smooth = []
    for c in comps:
        cs = bayesian_rating(c.get("rating"), c.get("user_ratings_total"), m_prior=200, C_prior=4.3)
        if cs is not None:
            comp_smooth.append(cs)
    if comp_smooth and smoothed is not None:
        avg_comp = sum(comp_smooth) / len(comp_smooth)
        # If you match or beat the average you get closer to 1; else scaled down
        diff = smoothed - avg_comp  # could be negative
        # Map ~[-1.0, +1.0] to [0,1] with a soft curve
        competition_raw = max(0.0, min(1.0, 0.5 + diff / 2.0))
    else:
        competition_raw = None

    # -------------------------
    # Weights & rescale if missing
    # -------------------------
    weights = {
        "performance": 0.30,
        "gbp_quality": 0.45,
        "category_match": 0.10,
        "competition_gap": 0.15,
    }

    components = [
        {
            "key": "performance",
            "weight": weights["performance"],
            "raw": round(perf_norm, 3) if psi_score is not None else None,
            "explain": "PageSpeed (mobile PERFORMANCE category)",
        },
        {
            "key": "gbp_quality",
            "weight": weights["gbp_quality"],
            "raw": round(gbp_quality_raw, 3) if gbp_err is None else None,
            "explain": "GBP rating/reviews/presence/open_now",
        },
        {
            "key": "category_match",
            "weight": weights["category_match"],
            "raw": round(cat_match_raw, 3) if cat_match_raw is not None else None,
            "explain": "GBP types vs. provided target_category",
        },
        {
            "key": "competition_gap",
            "weight": weights["competition_gap"],
            "raw": round(competition_raw, 3) if competition_raw is not None else None,
            "explain": "Your smoothed rating vs nearby competitors’ smoothed avg",
        },
    ]

    available_weight = sum(c["weight"] for c in components if c["raw"] is not None)
    # rescale so available parts sum to 1.0
    if available_weight <= 0:
        total_score = 0
        points_list = []
    else:
        for c in components:
            if c["raw"] is None:
                c["points"] = None
            else:
                w = c["weight"] / available_weight
                c["points"] = round(c["raw"] * (w * 100.0), 1)
                c["weight"] = round(w, 3)
        total_score = int(round(sum(c["points"] or 0.0 for c in components)))

    # Prepare output
    # weight displayed as percentages
    for c in components:
        c["weight"] = c["weight"] if isinstance(c["weight"], float) else c["weight"]
        # Pretty print weight as percent (frontends often show nicely; leaving numeric here)

    return {
        "input": {
            "business_name": business_name,
            "location": location,
            "website": website,
            "target_category": target_category,
        },
        "pagespeed": {
            "performance_score": int(psi_score) if psi_score is not None else None,
            "error": psi_err,
        },
        "gbp": {
            "found": biz.get("found"),
            "name": biz.get("name"),
            "rating": rating,
            "user_ratings_total": n_reviews,
            "open_now": open_now,
            "categories": categories,
            "website": biz.get("website"),
            "google_maps_url": biz.get("google_maps_url"),
        },
        "competitors_used": comps[:5],  # short summary for “see details”
        "components": components,
        "score": total_score,
        "warnings": None,
    }
