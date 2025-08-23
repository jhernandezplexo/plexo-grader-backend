from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import requests
import os
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ---------------------------
# FastAPI app + CORS
# ---------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-frontend.vercel.app",  # replace later with real domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Health
# ---------------------------
@app.get("/")
def root():
    return {"message": "Backend is live!"}

# ---------------------------
# /scan (PageSpeed Insights)
# ---------------------------
@app.get("/scan", summary="Lead")
def scan(
    name: str = Query(...),
    email: str = Query(...),
    business_name: str = Query(...),
    website: str = Query(...),
):
    if not website.startswith("http"):
        website = "https://" + website

    # Optional: forward to n8n
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
            pass  # non‑fatal

    api_key = os.getenv("PAGESPEED_API_KEY")
    psi_endpoint = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"

    params = {"url": website, "category": "PERFORMANCE", "strategy": "mobile"}
    if api_key:
        params["key"] = api_key

    performance_score = None
    psi_error = None

    try:
        r = requests.get(psi_endpoint, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        performance_score = round(
            data["lighthouseResult"]["categories"]["performance"]["score"] * 100
        )
    except Exception as e:
        try:
            psi_error = r.json().get("error", {}).get("message")
        except Exception:
            psi_error = str(e)

    return {
        "name": name,
        "email": email,
        "business_name": business_name,
        "website": website,
        "performance_score": performance_score,
        "psi_error": psi_error,
    }

# ---------------------------
# /business (GBP + competitors)
# ---------------------------
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

def _places_text_search(query: str) -> Dict[str, Any]:
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": GOOGLE_PLACES_API_KEY}
    return requests.get(url, params=params, timeout=20).json()

def _place_details(place_id: str) -> Dict[str, Any]:
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "key": GOOGLE_PLACES_API_KEY,
        "fields": "place_id,name,formatted_address,website,geometry,opening_hours,types,"
                  "rating,user_ratings_total,url"
    }
    return requests.get(url, params=params, timeout=20).json()

def _nearby_competitors(lat: float, lng: float, exclude_place_id: str) -> List[Dict[str, Any]]:
    """
    Simple nearby competitors: restaurants within ~1500 m, exclude the found place.
    We take top 5 by rating then reviews.
    """
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "key": GOOGLE_PLACES_API_KEY,
        "location": f"{lat},{lng}",
        "radius": 1500,
        "type": "restaurant",
        # no keyword filter so we get general local competitors
    }
    data = requests.get(url, params=params, timeout=20).json()

    items = []
    for r in data.get("results", []):
        if r.get("place_id") == exclude_place_id:
            continue
        items.append({
            "place_id": r.get("place_id"),
            "name": r.get("name"),
            "rating": r.get("rating"),
            "user_ratings_total": r.get("user_ratings_total"),
            "vicinity": r.get("vicinity"),
        })

    # Sort: rating desc, then review count desc
    items.sort(key=lambda x: (x.get("rating") or 0, x.get("user_ratings_total") or 0), reverse=True)
    return items[:5]

@app.get("/business")
def business(
    business_name: str = Query(..., description="e.g., 'Il Lago Trattoria'"),
    location: str = Query(..., description="City/State/Country, e.g., 'Doral, FL'"),
):
    if not GOOGLE_PLACES_API_KEY:
        return {
            "found": False,
            "error": "GOOGLE_PLACES_API_KEY is not set in environment.",
        }

    # 1) Find the business by text search
    query = f"{business_name} {location}"
    ts = _places_text_search(query)

    if ts.get("status") != "OK" or not ts.get("results"):
        return {"found": False, "error": ts.get("status") or "REQUEST_DENIED"}

    first = ts["results"][0]
    place_id = first["place_id"]

    # 2) Get details (addr, website, geometry, rating)
    det = _place_details(place_id)
    if det.get("status") != "OK":
        return {"found": False, "error": det.get("status")}

    details = det.get("result", {})
    name = details.get("name")
    formatted_address = details.get("formatted_address")
    website = details.get("website")
    rating = details.get("rating")
    user_ratings_total = details.get("user_ratings_total")
    maps_url = details.get("url")
    types = details.get("types") or []
    open_now = None
    try:
        open_now = details.get("opening_hours", {}).get("open_now")
    except Exception:
        pass

    lat = details.get("geometry", {}).get("location", {}).get("lat")
    lng = details.get("geometry", {}).get("location", {}).get("lng")

    # 3) Competitors nearby (top 5)
    competitors: List[Dict[str, Any]] = []
    if lat is not None and lng is not None:
        try:
            competitors = _nearby_competitors(lat, lng, place_id)
        except Exception as e:
            competitors = []
    
    return {
        "found": True,
        "place_id": place_id,
        "name": name,
        "formatted_address": formatted_address,
        "rating": rating,
        "user_ratings_total": user_ratings_total,
        "website": website,
        "google_maps_url": maps_url,
        "categories": types,
        "open_now": open_now,
        "competitors": competitors,
        "error": None,
    }

# ---------------------------
# /leads (Sheets + n8n)
# ---------------------------
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
            "private_key": (os.getenv("GOOGLE_PRIVATE_KEY") or "").replace("\\n", "\n"),
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
    if not n8n_url:
        return
    try:
        requests.post(n8n_url, json=lead.dict(), timeout=10)
    except requests.RequestException:
        pass
