from typing import Optional, List, Dict
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import re

app = FastAPI()

# --- CORS: add your real Vercel domain when you deploy frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend is live!"}

# =========================
# /scan (PageSpeed Insights)
# =========================
@app.get("/scan", summary="Lead")
def scan(
    name: str = Query(...),
    email: str = Query(...),
    business_name: str = Query(...),
    website: str = Query(...),
):
    # Normalize URL
    if not website.startswith("http"):
        website = "https://" + website

    # (Optional) forward the lead to n8n
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

    # PageSpeed Insights
    api_key = os.getenv("PAGESPEED_API_KEY")  # optional but recommended
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

# =================
# /leads (POST)
# =================
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

# =========================
# /business (GBP + Competitors)
# =========================

PLACES_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
if not PLACES_KEY:
    print("WARNING: GOOGLE_PLACES_API_KEY is not set")

TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

# Simple cuisine keyword map (extend as needed)
CUISINE_MAP = {
    "italian": ["italian", "trattoria", "osteria", "ristorante", "pizzeria", "pizza"],
    "mexican": ["mexican", "taqueria", "taco"],
    "japanese": ["japanese", "sushi", "ramen", "izakaya"],
    "chinese": ["chinese", "szechuan", "dim sum"],
    "thai": ["thai"],
    "indian": ["indian", "tandoori", "curry"],
    "french": ["french", "bistro", "brasserie"],
    "greek": ["greek", "taverna"],
    "peruvian": ["peruvian", "ceviche", "nikkei"],
    "korean": ["korean", "bbq"],
    "mediterranean": ["mediterranean"],
    "spanish": ["spanish", "tapas"],
    "lebanese": ["lebanese"],
    # …add more
}

def infer_cuisine_from_name(name: str) -> Optional[str]:
    n = name.lower()
    for cuisine, words in CUISINE_MAP.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", n):
                return cuisine
    return None

def contains_cuisine(name: str, cuisine: Optional[str]) -> bool:
    if not cuisine:
        return True
    n = name.lower()
    for w in CUISINE_MAP.get(cuisine, []):
        if w in n:
            return True
    return False

@app.get("/business")
def business(
    business_name: str = Query(..., description="Business name, e.g. 'Il Lago Trattoria'"),
    location: str = Query(..., description="City/State/Country, e.g. 'Doral, FL'"),
):
    if not PLACES_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_PLACES_API_KEY not configured")

    # 1) Find the business (Text Search)
    ts_params = {"query": f"{business_name} {location}", "key": PLACES_KEY}
    ts_res = requests.get(TEXT_SEARCH_URL, params=ts_params, timeout=20)
    ts_data = ts_res.json()

    if ts_data.get("status") != "OK" or not ts_data.get("results"):
        return {
            "found": False,
            "place_id": None,
            "name": None,
            "formatted_address": None,
            "rating": None,
            "user_ratings_total": None,
            "website": None,
            "google_maps_url": None,
            "categories": [],
            "open_now": None,
            "competitors": [],
            "error": ts_data.get("status", "NOT_FOUND"),
        }

    me = ts_data["results"][0]
    place_id = me.get("place_id")
    lat = me["geometry"]["location"]["lat"]
    lng = me["geometry"]["location"]["lng"]

    # Guess cuisine from the entered business name
    cuisine = infer_cuisine_from_name(business_name)

    # 2) Nearby competitors (type=restaurant, keyword=cuisine if available)
    nearby_params = {
        "location": f"{lat},{lng}",
        "radius": 2000,                 # ~2km
        "type": "restaurant",
        "key": PLACES_KEY,
    }
    if cuisine:
        nearby_params["keyword"] = cuisine

    nb_res = requests.get(NEARBY_URL, params=nearby_params, timeout=20)
    nb_data = nb_res.json()
    nb_results = nb_data.get("results", [])

    # Filter: exclude the same place, and keep only those matching cuisine in the name if cuisine detected
    competitors: List[Dict] = []
    for r in nb_results:
        if r.get("place_id") == place_id:
            continue
        name = r.get("name", "")
        if cuisine and not contains_cuisine(name, cuisine):
            continue
        competitors.append(
            {
                "place_id": r.get("place_id"),
                "name": name,
                "rating": r.get("rating"),
                "user_ratings_total": r.get("user_ratings_total"),
                "vicinity": r.get("vicinity"),
            }
        )

    # Rank by rating desc, then reviews desc; keep top 5
    competitors.sort(key=lambda x: (x.get("rating") or 0, x.get("user_ratings_total") or 0), reverse=True)
    competitors = competitors[:5]

    # Build primary categories (best-effort from Text Search 'types')
    categories = me.get("types", [])
    gm_url = f"https://www.google.com/maps/place/?q=place_id:{place_id}"

    return {
        "found": True,
        "place_id": place_id,
        "name": me.get("name"),
        "formatted_address": me.get("formatted_address"),
        "rating": me.get("rating"),
        "user_ratings_total": me.get("user_ratings_total"),
        "website": me.get("website", None),  # Text Search often doesn't include website; leaving for consistency
        "google_maps_url": gm_url,
        "categories": categories,
        "open_now": me.get("opening_hours", {}).get("open_now") if me.get("opening_hours") else None,
        "competitor_filter": cuisine,   # <-- shows which cuisine filter was used
        "competitors": competitors,
        "error": nb_data.get("status"),
    }
