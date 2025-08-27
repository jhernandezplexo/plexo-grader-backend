# main.py
# Plexo Grader Backend (FastAPI)

import os
import math
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Plexo Grader Backend", version="0.3.1")

# -----------------------------
# Environment
# -----------------------------
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "").strip()
PAGESPEED_API_KEY: str = os.getenv("PAGESPEED_API_KEY", "").strip()

# -----------------------------
# Constants
# -----------------------------
PLACES_TEXT_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
PLACES_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

# Reasonable default radius for "nearby competitors" (meters)
DEFAULT_RADIUS_M = 3000
# Max competitors returned
MAX_COMPETITORS = 10
# Optional light filter knobs
MIN_USER_RATINGS = 30  # ignore places with tiny number of ratings (often noise)
MIN_RATING = 3.6

# A simple synonyms table to tighten category matching
# (left side = normalized target category; right side = any words that imply it)
CATEGORY_SYNONYMS = {
    "restaurant": {"restaurant", "food", "meal_takeaway", "meal_delivery"},
    "italian": {"italian"},
    "pizza": {"pizza", "pizzeria"},
    "sushi": {"sushi", "japanese"},
    "japanese": {"japanese", "sushi"},
    "mexican": {"mexican", "taco"},
    "chinese": {"chinese"},
    "thai": {"thai"},
    "indian": {"indian"},
    "steakhouse": {"steakhouse", "steak"},
    "seafood": {"seafood"},
    "peruvian": {"peruvian"},
    "lebanese": {"lebanese"},
    "mediterranean": {"mediterranean", "greek", "turkish"},
    "bbq": {"barbecue", "bbq"},
    "burger": {"burger"},
    "vegan": {"vegan", "vegetarian"},
    "bakery": {"bakery"},
}

# Types that often appear but are too generic or misleading for competitor filtering
GENERIC_TYPES = {
    "point_of_interest",
    "establishment",
    "food",
    "store",
    "premise",
    "place_of_worship",
    "finance",
    "lodging",
    "tourist_attraction",
}

# -----------------------------
# Models
# -----------------------------
class Competitor(BaseModel):
    place_id: str
    name: str
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    vicinity: Optional[str] = None
    types: List[str] = Field(default_factory=list)
    google_maps_url: Optional[str] = None
    distance_m: Optional[int] = None  # distance from the main place (if computed)


class BusinessOut(BaseModel):
    found: bool = False
    place_id: Optional[str] = None
    name: Optional[str] = None
    formatted_address: Optional[str] = None
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    website: Optional[str] = None
    google_maps_url: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    open_now: Optional[bool] = None
    competitors: List[Competitor] = Field(default_factory=list)
    error: Optional[str] = None


class ScoreOut(BaseModel):
    ok: bool = True
    score: Optional[int] = None  # 0-100 for now (very simple)
    notes: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


# -----------------------------
# Helpers
# -----------------------------
def _ok(v: Optional[str]) -> bool:
    return bool(v and v.strip())


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Distance (meters) between two lat/lng points using the haversine formula.
    """
    R = 6371_000.0  # meters
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(d_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _norm_category(cat: Optional[str]) -> Optional[str]:
    if not cat:
        return None
    cat = cat.lower().strip()
    # pick the normalized key that contains this word
    for key, words in CATEGORY_SYNONYMS.items():
        if cat in words or key == cat:
            return key
    # fallback: keep a single word (e.g., "italian restaurant" -> "italian")
    return cat.split()[0]


def _types_to_categories(types: List[str]) -> List[str]:
    """
    Convert Google types to simplified categories for display & matching.
    """
    cats = set()
    lower = [t.lower() for t in types]
    for key, words in CATEGORY_SYNONYMS.items():
        if any(word in lower for word in words):
            cats.add(key)
    # always track 'restaurant' if 'restaurant' is present
    if "restaurant" in lower:
        cats.add("restaurant")
    return sorted(cats)


def _category_match(target: Optional[str], place_types: List[str]) -> bool:
    """
    Return True if the place_types look like a restaurant of the same cuisine/category (if provided).
    """
    lower = set(t.lower() for t in place_types)

    # must be a restaurant-like
    if "restaurant" not in lower:
        return False

    # filter out obviously generic-only entries
    if lower.issubset(GENERIC_TYPES | {"restaurant"}):
        # nothing specific besides "restaurant"
        return True if not target else False

    if not target:
        # no target => any restaurant is fine
        return True

    target = _norm_category(target)
    if not target:
        return True

    synonyms = CATEGORY_SYNONYMS.get(target, {target})
    # if any synonym appears in the place types, we accept
    if any(word in lower for word in synonyms):
        return True

    # For cases where Google's types don't include the cuisine word,
    # try a fuzzy-ish fallback: allow if the name contains the target word.
    return False


def _clean_competitor_types(types: Optional[List[str]]) -> List[str]:
    if not types:
        return []
    return [t for t in types if t not in GENERIC_TYPES]


def _details_maps_url(place_id: str) -> str:
    return f"https://maps.google.com/?cid={place_id}"


# -----------------------------
# Google Places (async httpx)
# -----------------------------
async def places_text_search(
    client: httpx.AsyncClient, query: str
) -> Optional[Dict[str, Any]]:
    if not _ok(GOOGLE_API_KEY):
        return None

    params = {"query": query, "key": GOOGLE_API_KEY}
    r = await client.get(PLACES_TEXT_URL, params=params, timeout=30)
    data = r.json()
    results = data.get("results") or []
    return results[0] if results else None


async def place_details(
    client: httpx.AsyncClient, place_id: str
) -> Optional[Dict[str, Any]]:
    if not _ok(GOOGLE_API_KEY):
        return None

    fields = (
        "place_id,name,formatted_address,types,rating,user_ratings_total,"
        "opening_hours,opening_hours/open_now,website,url,"
        "geometry/location,geometry,utc_offset_minutes"
    )
    params = {"place_id": place_id, "fields": fields, "key": GOOGLE_API_KEY}
    r = await client.get(PLACES_DETAILS_URL, params=params, timeout=30)
    data = r.json()
    return (data or {}).get("result")


async def nearby_restaurants(
    client: httpx.AsyncClient,
    lat: float,
    lng: float,
    radius_m: int = DEFAULT_RADIUS_M,
    page_limit: int = 2,
) -> List[Dict[str, Any]]:
    """
    Get nearby restaurants. We keep it simple: type=restaurant + radius.
    We page at most twice to control cost/latency.
    """
    all_results: List[Dict[str, Any]] = []
    token: Optional[str] = None
    pages = 0

    while True:
        params = {
            "location": f"{lat},{lng}",
            "radius": radius_m,
            "type": "restaurant",
            "key": GOOGLE_API_KEY,
        }
        if token:
            params["pagetoken"] = token

        r = await client.get(PLACES_NEARBY_URL, params=params, timeout=30)
        data = r.json()
        results = data.get("results") or []
        all_results.extend(results)

        token = data.get("next_page_token")
        pages += 1
        if not token or pages >= page_limit:
            break

    return all_results


# -----------------------------
# Public Endpoints
# -----------------------------
@app.get("/business", response_model=BusinessOut)
async def business(
    business_name: str = Query(..., description="Business name to search"),
    location: str = Query(..., description="City or area (e.g., Doral)"),
    website: Optional[str] = Query(
        None, description="Optional website (used only for echoing)"
    ),
    target_category: Optional[str] = Query(
        None,
        description="Strong hint for cuisine/category (e.g., 'Italian', 'Sushi').",
    ),
    radius_m: int = Query(DEFAULT_RADIUS_M, ge=500, le=10000),
):
    """
    Return main business info + filtered competitors:
      - Close by (within radius)
      - Is a restaurant
      - If target_category provided, must match that cuisine/category
      - Light denoising by rating & number of ratings
    """
    if not _ok(GOOGLE_API_KEY):
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_API_KEY is not set in the environment.",
        )

    query = f"{business_name} {location}"
    out = BusinessOut(found=False, competitors=[])

    async with httpx.AsyncClient() as client:
        main_candidate = await places_text_search(client, query)
        if not main_candidate:
            out.error = "No business found"
            return out

        place_id = main_candidate.get("place_id")
        details = await place_details(client, place_id=place_id)
        if not details:
            out.error = "No details available"
            return out

        # Extract main business basics
        out.found = True
        out.place_id = details.get("place_id")
        out.name = details.get("name")
        out.formatted_address = details.get("formatted_address")
        out.rating = details.get("rating")
        out.user_ratings_total = details.get("user_ratings_total")
        out.website = details.get("website") or website
        out.google_maps_url = details.get("url")
        dtype_list = details.get("types") or []
        out.categories = _types_to_categories(dtype_list)
        opening = (details.get("opening_hours") or {}).get("open_now")
        if isinstance(opening, bool):
            out.open_now = opening

        # location for radius filtering
        geom = (details.get("geometry") or {}).get("location") or {}
        lat, lng = geom.get("lat"), geom.get("lng")
        if lat is None or lng is None:
            # If geometry is missing, just skip competitors gracefully
            return out

        # Compute competitors
        near = await nearby_restaurants(client, lat, lng, radius_m=radius_m)

        # Choose matching category
        normalized_target = _norm_category(target_category)

        competitors: List[Competitor] = []
        for row in near:
            # Skip itself
            if row.get("place_id") == out.place_id:
                continue

            r_types = row.get("types") or []
            # 1) Must be a restaurant, 2) must match target if given
            if not _category_match(normalized_target, r_types):
                continue

            # Light denoising
            ur = row.get("user_ratings_total") or 0
            rating = row.get("rating") or 0.0
            if ur < MIN_USER_RATINGS or rating < MIN_RATING:
                continue

            # Compute distance (if both have geometry)
            loc = (row.get("geometry") or {}).get("location") or {}
            rlat, rlng = loc.get("lat"), loc.get("lng")
            dist_m = None
            if rlat is not None and rlng is not None:
                dist_m = int(round(_haversine_m(lat, lng, rlat, rlng)))

            comp = Competitor(
                place_id=row.get("place_id"),
                name=row.get("name"),
                rating=rating,
                user_ratings_total=ur,
                vicinity=row.get("vicinity"),
                types=_clean_competitor_types(r_types),
                google_maps_url=_details_maps_url(row.get("place_id")),
                distance_m=dist_m,
            )
            competitors.append(comp)

        # Sort: (1) closest, (2) more ratings, (3) higher rating
        competitors.sort(
            key=lambda c: (
                99_999 if c.distance_m is None else c.distance_m,
                -(c.user_ratings_total or 0),
                -(c.rating or 0),
            )
        )
        out.competitors = competitors[:MAX_COMPETITORS]
        return out


@app.get("/score", response_model=ScoreOut)
async def score(
    business_name: str = Query(...),
    location: str = Query(...),
    website: Optional[str] = None,
    target_category: Optional[str] = None,
    radius_m: int = Query(DEFAULT_RADIUS_M, ge=500, le=10000),
):
    """
    Very lightweight sample “score”:
      - Base = main rating scaled to 0..100
      - Minus small penalties if many nearby competitors with higher rating
    """
    try:
        biz = await business(
            business_name=business_name,
            location=location,
            website=website,
            target_category=target_category,
            radius_m=radius_m,
        )
        if not biz.found:
            return ScoreOut(ok=False, score=None, notes="Business not found")

        base = 0
        if biz.rating is not None:
            # Map 0..5 to 0..100
            base = max(0, min(100, int(round((biz.rating / 5.0) * 100))))

        stronger = 0
        for c in biz.competitors:
            if (c.rating or 0) > (biz.rating or 0):
                stronger += 1

        penalty = min(20, stronger * 3)  # up to -20
        total = max(0, min(100, base - penalty))

        return ScoreOut(
            ok=True,
            score=total,
            notes="Simple score based on rating and nearby stronger competitors.",
            raw={"base_from_rating": base, "stronger_competitors": stronger},
        )
    except Exception as e:
        return ScoreOut(ok=False, score=None, notes=f"Error: {e}")
