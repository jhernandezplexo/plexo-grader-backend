from typing import Optional, List, Dict, Any
from fastapi import Query
import os
import requests

# ---------- UPGRADED: GET /business ----------
@app.get("/business")
def business_lookup(
    business_name: str = Query(..., description="Business name, e.g. 'Il Lago Trattoria'"),
    location: str = Query(..., description="City/State/Country, e.g. 'Doral, FL'"),
):
    """
    Finds a business using Google Places Text Search + Details,
    and also returns up to 5 nearby competitor restaurants (Nearby Search).
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
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
            "error": "Missing GOOGLE_PLACES_API_KEY",
            "competitors": [],
        }

    # 1) Text Search to find the business
    text_search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    query = f"{business_name} {location}"
    try:
        ts = requests.get(text_search_url, params={"query": query, "key": api_key}, timeout=20)
        ts.raise_for_status()
        ts_data = ts.json()
    except Exception as e:
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
            "error": f"Text Search failed: {str(e)}",
            "competitors": [],
        }

    results = ts_data.get("results", [])
    if not results:
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
            "error": "REQUEST_DENIED or ZERO_RESULTS",
            "competitors": [],
        }

    primary = results[0]
    place_id = primary.get("place_id")
    name = primary.get("name")
    formatted_address = primary.get("formatted_address")
    rating = primary.get("rating")
    user_ratings_total = primary.get("user_ratings_total")
    types = primary.get("types", [])
    open_now = None
    try:
        open_now = primary.get("opening_hours", {}).get("open_now")
    except Exception:
        pass

    # 2) Details for website (and we can create a Maps URL)
    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
    website = None
    google_maps_url = f"https://www.google.com/maps/place/?q=place_id:{place_id}" if place_id else None

    try:
        det = requests.get(
            details_url,
            params={
                "place_id": place_id,
                "fields": "website,opening_hours",  # keep it light for quota
                "key": api_key,
            },
            timeout=20,
        )
        det.raise_for_status()
        det_data = det.json().get("result", {})
        website = det_data.get("website") or website
        # opening hours may be richer here
        if open_now is None:
            try:
                open_now = det_data.get("opening_hours", {}).get("open_now")
            except Exception:
                pass
    except Exception:
        pass  # if details fails, we still return the text-search info

    # 3) Nearby competitors (type=restaurant, radius 1500m), exclude our own place_id
    competitors: List[Dict[str, Any]] = []
    try:
        geometry = primary.get("geometry", {})
        loc = geometry.get("location", {})
        lat, lng = loc.get("lat"), loc.get("lng")

        if lat is not None and lng is not None:
            nearby_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            nb = requests.get(
                nearby_url,
                params={
                    "location": f"{lat},{lng}",
                    "radius": 1500,             # ~1.5 km
                    "type": "restaurant",
                    "key": api_key,
                },
                timeout=20,
            )
            nb.raise_for_status()
            nb_data = nb.json()

            # Sort by user_ratings_total desc (a simple relevance proxy)
            raw = nb_data.get("results", [])
            raw = [r for r in raw if r.get("place_id") != place_id]
            raw.sort(key=lambda r: (r.get("user_ratings_total") or 0), reverse=True)

            for r in raw[:5]:
                competitors.append({
                    "place_id": r.get("place_id"),
                    "name": r.get("name"),
                    "rating": r.get("rating"),
                    "user_ratings_total": r.get("user_ratings_total"),
                    "vicinity": r.get("vicinity"),
                    "google_maps_url": f"https://www.google.com/maps/place/?q=place_id:{r.get('place_id')}",
                })
    except Exception:
        # ignore competitor errors; still return the primary record
        pass

    return {
        "found": True,
        "place_id": place_id,
        "name": name,
        "formatted_address": formatted_address,
        "rating": rating,
        "user_ratings_total": user_ratings_total,
        "website": website,
        "google_maps_url": google_maps_url,
        "categories": types or [],
        "open_now": open_now,
        "error": None,
        "competitors": competitors,
    }
