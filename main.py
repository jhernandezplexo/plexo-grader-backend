from typing import Optional, List, Dict
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os

app = FastAPI()

# ----- CORS -----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://your-frontend.vercel.app",  # replace with your real Vercel domain when you have it
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Root -----
@app.get("/")
def root():
    return {"message": "Backend is live!"}

# ========================
#  GET /scan  (PageSpeed)
# ========================
@app.get("/scan", summary="Lead")
def scan(
    name: str = Query(...),
    email: str = Query(...),
    business_name: str = Query(...),
    website: str = Query(...),
):
    if not website.startswith("http"):
        website = "https://" + website

    # Optional webhook to n8n
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

    # PageSpeed Insights
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

# =========================================
#  GET /business  (GBP + Competitors Filter)
# =========================================
@app.get("/business")
def business(
    business_name: str = Query(..., description="Business name, e.g. 'Il Lago Trattoria'"),
    location: str = Query(..., description="City/State/Country, e.g. 'Doral, FL'"),
):
    """
    Looks up the business in Google Places, detects its most specific restaurant type,
    then finds nearby competitors with the SAME type (e.g., italian_restaurant).
    """
    places_key = os.getenv("GOOGLE_PLACES_API_KEY")
    if not places_key:
        raise HTTPException(status_code=500, detail="Missing GOOGLE_PLACES_API_KEY")

    # --- 1) Text Search to find the business + place_id ---
    try:
        text_q = f"{business_name} {location}"
        ts_resp = requests.get(
            "https://maps.googleapis.com/maps/api/place/textsearch/json",
            params={"query": text_q, "key": places_key},
            timeout=10,
        )
        ts = ts_resp.json()
        if ts.get("status") != "OK" or not ts.get("results"):
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
                "error": ts.get("status"),
            }

        hit = ts["results"][0]
        place_id = hit["place_id"]
        lat = hit["geometry"]["location"]["lat"]
        lng = hit["geometry"]["location"]["lng"]
        formatted_address = hit.get("formatted_address")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Places textsearch failed: {e}")

    # --- 2) Place Details to enrich fields + types ---
    try:
        det_resp = requests.get(
            "https://maps.googleapis.com/maps/api/place/details/json",
            params={
                "place_id": place_id,
                "key": places_key,
                "fields": "name,website,url,rating,user_ratings_total,opening_hours,types",
            },
            timeout=10,
        )
        det = det_resp.json()
        if det.get("status") != "OK":
            return {
                "found": True,
                "place_id": place_id,
                "name": hit.get("name"),
                "formatted_address": formatted_address,
                "rating": None,
                "user_ratings_total": None,
                "website": None,
                "google_maps_url": None,
                "categories": [],
                "open_now": None,
                "competitors": [],
                "error": det.get("status"),
            }

        details = det.get("result", {})
        categories: List[str] = details.get("types", []) or hit.get("types", []) or []

        # Choose the most specific restaurant-like type (e.g., italian_restaurant).
        # Fallback to 'restaurant' if no cuisine-specific type exists.
        primary_type = "restaurant"
        for t in categories:
            if t.endswith("_restaurant") and t != "restaurant":
                primary_type = t
                break
        if primary_type == "restaurant" and "restaurant" not in categories:
            # still ensure it's a restaurant category; if not, just leave as 'restaurant'
            pass

        # --- 3) Nearby competitors filtered by 'primary_type' ---
        # Note: we keep a reasonable radius (e.g. 3000m). Adjust as needed.
        nearby_resp = requests.get(
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
            params={
                "location": f"{lat},{lng}",
                "radius": 3000,
                "type": primary_type,
                "key": places_key,
            },
            timeout=10,
        )
        nearby = nearby_resp.json()
        comps_raw: List[Dict] = nearby.get("results", [])
        # Exclude the business itself; take top 5
        competitors: List[Dict] = []
        for r in comps_raw:
            if r.get("place_id") == place_id:
                continue
            competitors.append(
                {
                    "place_id": r.get("place_id"),
                    "name": r.get("name"),
                    "rating": r.get("rating"),
                    "user_ratings_total": r.get("user_ratings_total"),
                    "vicinity": r.get("vicinity"),
                }
            )
            if len(competitors) == 5:
                break

        return {
            "found": True,
            "place_id": place_id,
            "name": details.get("name", hit.get("name")),
            "formatted_address": formatted_address,
            "rating": details.get("rating"),
            "user_ratings_total": details.get("user_ratings_total"),
            "website": details.get("website"),
            "google_maps_url": details.get("url"),
            "categories": categories,
            "open_now": (details.get("opening_hours") or {}).get("open_now"),
            "competitors": competitors,
            "error": None,
        }

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Places details/nearby failed: {e}")

# ======================
#  POST /leads (existing)
# ======================
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

# --------- Helpers (Sheets + n8n) ---------
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
            "private_key": os.getenv("GOOGLE_PRIVATE_KEY", "").replace("\\n", "\n"),
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
