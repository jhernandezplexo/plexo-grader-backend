from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import statistics
import math
import os

app = FastAPI()

# CORS (add your real Vercel domain when ready)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-frontend.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Root ----------
@app.get("/")
def root():
    return {"message": "Backend is live!"}

# =========================
# Helpers (shared)
# =========================

def fetch_pagespeed_performance(website: str) -> Dict[str, Any]:
    """Returns {'score': Optional[int], 'error': Optional[str]}"""
    if not website.startswith("http"):
        website = "https://" + website

    api_key = os.getenv("PAGESPEED_API_KEY")  # optional
    endpoint = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
    params = {"url": website, "category": "PERFORMANCE", "strategy": "mobile"}
    if api_key:
        params["key"] = api_key

    try:
        r = requests.get(endpoint, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        score = round(
            data["lighthouseResult"]["categories"]["performance"]["score"] * 100
        )
        return {"score": score, "error": None}
    except Exception as e:
        try:
            err = r.json().get("error", {}).get("message")
        except Exception:
            err = str(e)
        return {"score": None, "error": err}

def parse_target_category(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return s.strip().lower()

def string_contains_any(haystack: str, need: List[str]) -> bool:
    h = haystack.lower()
    return any(n in h for n in need)

def gbp_type_is_restaurant(cat: str) -> bool:
    return "restaurant" in cat

def call_places_textsearch(query: str, key: str) -> Dict[str, Any]:
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    return requests.get(url, params={"query": query, "key": key}, timeout=20).json()

def call_places_details(place_id: str, key: str) -> Dict[str, Any]:
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    fields = ",".join([
        "name",
        "rating",
        "user_ratings_total",
        "formatted_address",
        "website",
        "opening_hours",
        "types"
    ])
    return requests.get(url, params={"place_id": place_id, "fields": fields, "key": key}, timeout=20).json()

def call_places_nearby(lat: float, lng: float, key: str, radius_m: int = 2000) -> Dict[str, Any]:
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    return requests.get(
        url,
        params={
            "location": f"{lat},{lng}",
            "radius": radius_m,
            "type": "restaurant",
            "key": key,
        },
        timeout=20,
    ).json()

def extract_lat_lng_from_textsearch(result: Dict[str, Any]) -> Optional[Dict[str, float]]:
    try:
        g = result["geometry"]["location"]
        return {"lat": g["lat"], "lng": g["lng"]}
    except Exception:
        return None

def category_matches(target: Optional[str], types: List[str]) -> bool:
    """Loose match: if 'italian' in target -> look for 'italian' + 'restaurant'."""
    if not target:
        return False
    t = target.lower()
    # direct type hit (google types like 'restaurant', 'italian_restaurant', etc.)
    if t.replace(" ", "_") in types:
        return True
    # loose contains across types text
    joined = " ".join(types).lower()
    return t in joined

# =========================
# Existing: /scan (PageSpeed only)
# =========================
@app.get("/scan", summary="Lead")
def scan(
    name: str = Query(...),
    email: str = Query(...),
    business_name: str = Query(...),
    website: str = Query(...),
):
    if not website.startswith("http"):
        website = "https://" + website

    # Optional lead forwarding
    n8n_url = os.getenv("N8N_WEBHOOK_URL")
    if n8n_url:
        try:
            requests.post(
                n8n_url,
                json={
                    "name": name,
                    "email": email,
                    "business_name": business_name,
                    "website": website,
                },
                timeout=10,
            )
        except requests.RequestException:
            pass

    psi = fetch_pagespeed_performance(website)

    return {
        "name": name,
        "email": email,
        "business_name": business_name,
        "website": website,
        "performance_score": psi["score"],
        "psi_error": psi["error"],
    }

# =========================
# Existing: /business (GBP + competitors, with category filter)
# =========================
@app.get("/business")
def business(
    business_name: str = Query(..., description="Business name, e.g. 'Il Lago Trattoria'"),
    location: str = Query(..., description="City/State/Country, e.g. 'Doral, FL'"),
    target_category: Optional[str] = Query(None, description="e.g. 'Italian restaurant'"),
):
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing GOOGLE_PLACES_API_KEY")

    tcat = parse_target_category(target_category)

    # 1) Find the place
    search = call_places_textsearch(f"{business_name} {location}", api_key)
    if search.get("status") != "OK" or not search.get("results"):
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
            "error": search.get("status"),
        }

    primary = search["results"][0]
    place_id = primary["place_id"]

    # 2) Details
    details = call_places_details(place_id, api_key)
    if details.get("status") != "OK":
        return {
            "found": False,
            "place_id": place_id,
            "name": None,
            "formatted_address": None,
            "rating": None,
            "user_ratings_total": None,
            "website": None,
            "google_maps_url": f"https://maps.google.com/?cid={primary.get('place_id','')}",
            "categories": None,
            "open_now": None,
            "competitors": [],
            "error": details.get("status"),
        }

    result = details["result"]
    name = result.get("name")
    rating = result.get("rating")
    reviews = result.get("user_ratings_total")
    formatted_address = result.get("formatted_address")
    website = result.get("website")
    open_now = None
    if result.get("opening_hours") and "open_now" in result["opening_hours"]:
        open_now = result["opening_hours"]["open_now"]
    types = result.get("types", [])
    gmaps_url = f"https://maps.google.com/?cid={place_id}"

    # 3) Competitors (nearby restaurants)
    coords = extract_lat_lng_from_textsearch(primary)
    comp_list: List[Dict[str, Any]] = []
    if coords:
        nearby = call_places_nearby(coords["lat"], coords["lng"], api_key, radius_m=2000)
        for p in nearby.get("results", []):
            # Filter to restaurants only
            types_p = p.get("types", [])
            if not any(gbp_type_is_restaurant(t) for t in types_p):
                continue

            # Optional: filter to same cuisine / target category
            if tcat and not category_matches(tcat, types_p):
                continue

            comp_list.append({
                "place_id": p.get("place_id"),
                "name": p.get("name"),
                "rating": p.get("rating"),
                "user_ratings_total": p.get("user_ratings_total"),
                "vicinity": p.get("vicinity"),
            })

        # keep the strongest 5 by review count
        comp_list = sorted(
            [c for c in comp_list if c.get("rating") is not None],
            key=lambda x: (x.get("user_ratings_total") or 0),
            reverse=True
        )[:5]

    return {
        "found": True,
        "place_id": place_id,
        "name": name,
        "formatted_address": formatted_address,
        "rating": rating,
        "user_ratings_total": reviews,
        "website": website,
        "google_maps_url": gmaps_url,
        "categories": types,
        "open_now": open_now,
        "competitors": comp_list,
        "error": None,
    }

# =========================
# NEW: /score  (overall 0–100 with breakdown)
# =========================
@app.get("/score")
def score(
    business_name: str = Query(..., description="e.g. 'Il Lago Trattoria'"),
    location: str = Query(..., description="e.g. 'Doral, FL'"),
    website: str = Query(..., description="e.g. 'https://example.com'"),
    target_category: Optional[str] = Query(None, description="e.g. 'Italian restaurant'"),
):
    """
    Returns an overall score (0–100) and a per-component breakdown.
    Components (default weights):
      - performance (PageSpeed): 40%
      - gbp quality (rating, reviews, presence, open_now): 35%
      - category match (if target_category provided): 10%
      - competition gap (your rating vs competitors avg): 15%
    If any component is missing, weights auto-rescale to the signals we have.
    """
    warnings: List[str] = []

    # 1) PageSpeed
    psi = fetch_pagespeed_performance(website)
    perf_raw = None  # 0..1
    if psi["score"] is not None:
        perf_raw = max(0.0, min(1.0, psi["score"] / 100.0))
    else:
        warnings.append(f"PageSpeed error: {psi['error']}")

    # 2) GBP + competitors
    biz = business(business_name=business_name, location=location, target_category=target_category)
    if not biz.get("found"):
        warnings.append(f"Business not found or Places error: {biz.get('error')}")
    rating = biz.get("rating")
    reviews = biz.get("user_ratings_total")
    open_now = biz.get("open_now")
    types = biz.get("categories") or []
    comps: List[Dict[str, Any]] = biz.get("competitors") or []

    # 2a) GBP quality raw 0..1
    gbp_raw = None
    if rating is not None:
        rating_part = max(0.0, min(1.0, float(rating) / 5.0))
        # reviews: log-scale, cap at ~10k reviews
        # log10(10000)=4 → /4 ≈ 1.0 cap
        rev_part = 0.0
        if reviews is not None:
            rev_part = max(0.0, min(1.0, math.log10(max(1, int(reviews))) / 4.0))
        presence_part = 1.0  # we found it, so +presence
        open_part = 0.0
        if open_now is True:
            open_part = 0.2  # small bonus
        # Weighted inside gbp: rating 0.6, reviews 0.2, presence 0.15, open_now 0.05
        gbp_raw = max(0.0, min(1.0, (0.6 * rating_part) + (0.2 * rev_part) + (0.15 * presence_part) + (0.05 * open_part)))
    else:
        warnings.append("GBP rating unavailable; GBP component skipped.")

    # 2b) Category match raw 0..1 (only if target_category provided)
    cat_raw = None
    tcat = parse_target_category(target_category)
    if tcat:
        cat_raw = 1.0 if category_matches(tcat, types) else 0.0
        if cat_raw == 0.0:
            warnings.append(f"Target category '{tcat}' not detected in GBP types.")

    # 2c) Competition gap raw 0..1
    comp_raw = None
    if rating is not None and comps:
        comp_ratings = [c.get("rating") for c in comps if c.get("rating") is not None]
        if comp_ratings:
            comp_avg = statistics.mean(comp_ratings)
            diff = float(rating) - float(comp_avg)  # positive is good
            # map diff in [-0.5, +0.5] to [0..1] (clip outside)
            # 0.5 above avg => 1.0; 0.5 below => 0.0
            comp_raw = max(0.0, min(1.0, (diff + 0.5) / 1.0))
        else:
            warnings.append("No competitor ratings available; competition component skipped.")
    elif rating is None:
        warnings.append("Missing your GBP rating; competition component skipped.")
    else:
        warnings.append("No competitors found; competition component skipped.")

    # 3) Aggregate with adaptive weights
    # base weights
    W_PERF = 0.40
    W_GBP = 0.35
    W_CAT = 0.10
    W_COMP = 0.15

    parts = [
        {"key": "performance", "weight": W_PERF, "raw": perf_raw, "explain": "PageSpeed (mobile PERFORMANCE category)"},
        {"key": "gbp_quality", "weight": W_GBP, "raw": gbp_raw, "explain": "GBP rating/reviews/presence/open_now"},
        {"key": "category_match", "weight": W_CAT, "raw": cat_raw, "explain": "GBP types vs. provided target_category"},
        {"key": "competition_gap", "weight": W_COMP, "raw": comp_raw, "explain": "Your rating vs. nearby competitors’ avg"},
    ]

    available_weight = sum(p["weight"] for p in parts if p["raw"] is not None)
    if available_weight == 0:
        raise HTTPException(status_code=502, detail="No usable signals to compute a score.")

    # normalized 0..100
    total = sum((p["raw"] or 0.0) * p["weight"] for p in parts) / available_weight * 100.0
    total_rounded = round(total)

    # add points (each component’s contribution in 0..100 terms)
    for p in parts:
        if p["raw"] is None:
            p["points"] = None
        else:
            p["points"] = round((p["raw"] * p["weight"] / available_weight) * 100.0, 1)

    return {
        "input": {
            "business_name": business_name,
            "location": location,
            "website": website,
            "target_category": tcat,
        },
        "pagespeed": {"performance_score": psi["score"], "error": psi["error"]},
        "gbp": {
            "found": biz.get("found"),
            "name": biz.get("name"),
            "rating": rating,
            "user_ratings_total": reviews,
            "open_now": open_now,
            "categories": types,
            "competitors_count": len(comps),
        },
        "components": parts,
        "score": total_rounded,
        "warnings": warnings or None,
    }

# =========================
# Existing: POST /leads (unchanged)
# =========================
class Lead(BaseModel):
    name: str
    email: str
    business_name: str
    website: str

@app.post("/leads")
async def capture_lead(lead: Lead):
    store_lead_in_google_sheets(lead)
    trigger_email_automation(lead)
    return {"status": "success", "lead": lead}

def store_lead_in_google_sheets(lead: Lead):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(
        {
            "type": "service_account",
            "project_id": os.getenv("GOOGLE_PROJECT_ID"),
            "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
            "private_key": os.getenv("GOOGLE_PRIVATE_KEY").replace("\\n", "\n"),
            "client_email": os.getenv("GOOGLE_SERVICE_ACCOUNT_EMAIL"),
            "client_id": os.getenv("GOOGLE_CLIENT_ID"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_X509_URL"),
        },
        scope,
    )
    client = gspread.authorize(creds)
    sheet = client.open_by_key(os.getenv("GOOGLE_SHEET_ID")).sheet1
    sheet.append_row(
        [
            datetime.utcnow().isoformat(),
            lead.name,
            lead.email,
            lead.business_name,
            lead.website,
        ]
    )

def trigger_email_automation(lead: Lead):
    n8n_url = os.getenv("N8N_WEBHOOK_URL")
    if n8n_url:
        try:
            requests.post(n8n_url, json=lead.dict(), timeout=10)
        except requests.RequestException:
            pass
