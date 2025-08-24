# ---------- GET /score ----------
from math import log
from typing import Dict, Any, List, Optional

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def _bayes_rating(rating: Optional[float], reviews: Optional[int], C: float = 4.3, m: int = 200) -> Optional[float]:
    if rating is None or reviews is None:
        return None
    try:
        n = max(0, int(reviews))
        r = float(rating)
        return (m * C + n * r) / (m + n)
    except Exception:
        return None

def _normalize_reviews(n: Optional[int], cap: int = 5000) -> float:
    """
    0..1 review strength using a log curve, saturating ~5k reviews.
    """
    if not isinstance(n, (int, float)) or n <= 0:
        return 0.0
    return _clamp(log(1 + n) / log(1 + cap))

def _category_match_score(gbp_types: List[str], target_category: Optional[str]) -> Optional[float]:
    if not target_category:
        return None
    target = set(target_category.lower().replace(",", " ").split())
    if not gbp_types:
        return 0.0
    ours = set()
    for t in gbp_types:
        for w in t.lower().replace("_", " ").split():
            if w:
                ours.add(w)
    inter = len(target & ours)
    union = len(target | ours) or 1
    return _clamp(inter / union)

def _competition_gap_score(our_bayes: Optional[float], comp_bayes_list: List[float]) -> Optional[float]:
    """
    Convert (our_bayes - avg_comp_bayes) in roughly [-1.0, +1.0] star band into 0..1.
    If you’re 1★ above average → ~1.0; 1★ below → ~0.0; equal → 0.5.
    """
    if our_bayes is None or not comp_bayes_list:
        return None
    avg_comp = sum(comp_bayes_list) / len(comp_bayes_list)
    delta = our_bayes - avg_comp
    return _clamp(0.5 + 0.5 * (delta / 1.0))  # scale 1 star delta to full-range swing

def _gbp_quality_score(gbp: Dict[str, Any]) -> float:
    """
    GBP quality combines:
    - bayesian rating (0..1 after /5)
    - review strength (log curve)
    - presence signals (website/phone/hours) + small bonus if open_now
    Weighted internally then clamped 0..1.
    """
    rating = gbp.get("rating")
    reviews = gbp.get("user_ratings_total")
    bayes = _bayes_rating(rating, reviews)  # ~ 1..5
    bayes_norm = 0.0 if bayes is None else _clamp(bayes / 5.0)

    review_strength = _normalize_reviews(reviews)

    has_site = 1.0 if gbp.get("website") else 0.0
    has_phone = 1.0 if gbp.get("formatted_phone_number") or gbp.get("international_phone_number") else 0.0
    has_hours = 1.0 if gbp.get("opening_hours") or gbp.get("current_opening_hours") else 0.0
    presence = (has_site + has_phone + has_hours) / 3.0

    open_now = None
    if isinstance(gbp.get("opening_hours"), dict) and "open_now" in gbp["opening_hours"]:
        open_now = gbp["opening_hours"]["open_now"]
    elif isinstance(gbp.get("current_opening_hours"), dict) and "open_now" in gbp["current_opening_hours"]:
        open_now = gbp["current_opening_hours"]["open_now"]
    open_bonus = 0.05 if open_now is True else 0.0

    # internal weighting inside GBP quality
    gbp_raw = 0.60 * bayes_norm + 0.25 * review_strength + 0.15 * presence
    return _clamp(gbp_raw + open_bonus, 0.0, 1.0)

def _fetch_pagespeed_score(website: str, api_key: Optional[str]) -> (Optional[int], Optional[str]):
    psi_endpoint = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
    params = {"url": website, "category": "PERFORMANCE", "strategy": "mobile"}
    if api_key:
        params["key"] = api_key
    try:
        r = requests.get(psi_endpoint, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        perf = round(data["lighthouseResult"]["categories"]["performance"]["score"] * 100)
        return perf, None
    except Exception as e:
        try:
            return None, r.json().get("error", {}).get("message")
        except Exception:
            return None, str(e)

def _places_text_search(business_name: str, location: str, key: str) -> Optional[str]:
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{business_name}, {location}",
        "type": "restaurant",
        "key": key,
    }
    r = requests.get(url, params=params, timeout=20)
    j = r.json()
    if j.get("status") != "OK" or not j.get("results"):
        return None
    return j["results"][0]["place_id"]

def _place_details(place_id: str, key: str) -> Dict[str, Any]:
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    fields = ",".join([
        "name",
        "rating",
        "user_ratings_total",
        "formatted_address",
        "formatted_phone_number",
        "international_phone_number",
        "website",
        "opening_hours",
        "current_opening_hours",
        "types",
        "url",
        "business_status",
        "price_level",
        "vicinity",
        "photos",
    ])
    params = {"place_id": place_id, "fields": fields, "key": key}
    r = requests.get(url, params=params, timeout=20)
    j = r.json()
    return j.get("result", {}) if j.get("status") == "OK" else {}

def _find_competitors(
    place_id: str,
    gbp_types: List[str],
    lat: float,
    lng: float,
    key: str,
    radius_m: int = 3000,
    limit: int = 8,
    filter_to_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Nearby search + (optional) filter by our top-level cuisine keywords present in types.
    """
    nearby_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius_m,
        "type": "restaurant",
        "key": key,
    }
    r = requests.get(nearby_url, params=params, timeout=20)
    j = r.json()
    results = j.get("results", [])
    comps = []
    # cuisine filter words pulled from our types (e.g., 'italian', 'pizza')
    type_words = set()
    for t in gbp_types or []:
        for w in t.lower().replace("_", " ").split():
            if len(w) >= 4:  # skip tiny words
                type_words.add(w)
    for res in results:
        if res.get("place_id") == place_id:
            continue
        c_types = res.get("types", [])
        if filter_to_types:
            # require at least one desired type token
            ok = any(tt in c_types for tt in filter_to_types)
            if not ok:
                continue
        # or loose cuisine keyword match
        if type_words:
            joined = " ".join(c_types).lower().replace("_", " ")
            if not any(w in joined for w in type_words):
                continue
        comps.append({
            "place_id": res.get("place_id"),
            "name": res.get("name"),
            "rating": res.get("rating"),
            "user_ratings_total": res.get("user_ratings_total"),
            "vicinity": res.get("vicinity"),
            "types": c_types,
            "geometry": res.get("geometry"),
        })
        if len(comps) >= limit:
            break
    return comps

@app.get("/score", summary="Score")
def score(
    business_name: str = Query(...),
    location: str = Query(...),
    website: str = Query(...),
    target_category: Optional[str] = Query(None),
):
    # 1) PageSpeed
    api_key_psi = os.getenv("PAGESPEED_API_KEY")
    perf_score, psi_error = _fetch_pagespeed_score(website, api_key_psi)

    # 2) GBP lookups
    places_key = os.getenv("GOOGLE_PLACES_API_KEY")
    place_id = None
    gbp = {}
    comps: List[Dict[str, Any]] = []
    if places_key:
        try:
            place_id = _places_text_search(business_name, location, places_key)
            if place_id:
                gbp = _place_details(place_id, places_key) or {}
                # coords for competitor search
                lat = gbp.get("geometry", {}).get("location", {}).get("lat")
                lng = gbp.get("geometry", {}).get("location", {}).get("lng")
                if lat is not None and lng is not None:
                    comps = _find_competitors(
                        place_id,
                        gbp.get("types", []) or [],
                        float(lat),
                        float(lng),
                        places_key,
                        radius_m=3000,
                        limit=8,
                    )
        except Exception:
            pass

    # 3) Component raw scores 0..1 (some can be None)
    perf_raw = None if perf_score is None else _clamp(perf_score / 100.0)
    gbp_raw = _gbp_quality_score(gbp) if gbp else None
    cat_raw = _category_match_score(gbp.get("types", []) if gbp else [], target_category)

    our_bayes = _bayes_rating(gbp.get("rating"), gbp.get("user_ratings_total")) if gbp else None
    comp_bayes_list = []
    for c in comps:
        comp_b = _bayes_rating(c.get("rating"), c.get("user_ratings_total"))
        if comp_b is not None:
            c["bayes"] = comp_b
            comp_bayes_list.append(comp_b)
    comp_raw = _competition_gap_score(our_bayes, comp_bayes_list) if comp_bayes_list else None

    # 4) Weights (your “Local SEO heavier” setting)
    weights = {
        "performance": 0.30,
        "gbp_quality": 0.45,
        "category_match": 0.10,
        "competition_gap": 0.15,
    }

    # 5) Auto‑rescale if any pieces missing
    components = [
        ("performance", perf_raw,      "PageSpeed (mobile PERFORMANCE category)"),
        ("gbp_quality", gbp_raw,       "GBP rating/reviews/presence/open_now"),
        ("category_match", cat_raw,    "GBP types vs. provided target_category"),
        ("competition_gap", comp_raw,  "Your rating vs. nearby competitors’ avg (Bayesian)"),
    ]
    available_w = sum(weights[k] for k, raw, _ in components if raw is not None) or 1.0
    scale = 1.0 / available_w
    total_points = 0.0
    comp_rows = []
    for key, raw, explain in components:
        w = weights[key]
        if raw is None:
            comp_rows.append({
                "key": key,
                "weight": w,
                "raw": None,
                "explain": explain,
                "points": None,
            })
            continue
        pts = 100.0 * (w * scale) * raw
        total_points += pts
        comp_rows.append({
            "key": key,
            "weight": w,
            "raw": round(raw, 4),
            "explain": explain,
            "points": round(pts, 1),
        })

    # 6) Assemble competitor summary (small + ready to show)
    comp_summary = [{
        "name": c.get("name"),
        "rating": c.get("rating"),
        "reviews": c.get("user_ratings_total"),
        "bayes": round(c.get("bayes", 0.0), 3) if isinstance(c.get("bayes"), (int, float)) else None,
        "vicinity": c.get("vicinity"),
        "types": c.get("types"),
    } for c in comps]

    return {
        "input": {
            "business_name": business_name,
            "location": location,
            "website": website,
            "target_category": target_category,
        },
        "pagespeed": {
            "performance_score": perf_score,
            "error": psi_error,
        },
        "gbp": gbp or None,              # raw GBP object (expand in UI)
        "competitors": comp_summary,     # compact list for UI details
        "components": comp_rows,         # 4-row table (weight/raw/points/explain)
        "score": round(total_points),    # 0..100 number grade
        "warnings": None,
    }
