from typing import Optional, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os

app = FastAPI()

# -------------------------
# CORS (allow your frontend)
# -------------------------
# Tip: replace "https://your-frontend.vercel.app" with your real Vercel/Trickle URL when you have it.
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


# =========================================================
# 1) GET /scan  — Website performance score (PageSpeed API)
# =========================================================
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

    # (Optional) forward the lead to n8n if you want
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
            # Non-fatal; continue
            pass

    # PageSpeed Insights (returns 0..100 score)
    api_key = os.getenv("PAGESPEED_API_KEY")  # optional but recommended
    psi_endpoint = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
    params = {"url": website, "category": "PERFORMANCE", "strategy": "mobile"}
    if api_key:
        params["key"] = api_key

    performance_score: Optional[int] = None
    psi_error: Optional[str] = None

    try:
        r = requests.get(psi_endpoint, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        performance_score = round(
            data["lighthouseResult"]["categories"]["performance"]["score"] * 100
        )
    except Exception as e:
        try:
            # If API returned a structured error
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


# =========================================================
# 2) POST /leads — store in Google Sheets + trigger n8n
# =========================================================
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


# -----------------
# Google Sheets I/O
# -----------------
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


# =========================================================
# 3) GET /business — Google Places (Business Profile check)
# =========================================================
class BusinessResponse(BaseModel):
    found: bool
    place_id: Optional[str] = None
    name: Optional[str] = None
    formatted_address: Optional[str] = None
    rating: Optional[float] = None
    user_ratings_total: Optional[int] = None
    website: Optional[str] = None
    google_maps_url: Optional[str] = None
    categories: Optional[List[str]] = None
    open_now: Optional[bool] = None
    error: Optional[str] = None

@app.get("/business", response_model=BusinessResponse, summary="Get Google Business info")
def get_business(
    business_name: str = Query(..., description="Business name, e.g. 'Il Lago Trattoria'"),
    location: str = Query(..., description="City/State/Country, e.g. 'Tiburon, CA'"),
):
    """
    1) Find the place from business_name + location
    2) Get details by place_id
    """
    PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
    if not PLACES_API_KEY:
        return BusinessResponse(found=False, error="Server missing GOOGLE_PLACES_API_KEY")

    # 1) Find Place
    find_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
    find_params = {
        "input": f"{business_name}, {location}",
        "inputtype": "textquery",
        "fields": "place_id,name,formatted_address",
        "key": PLACES_API_KEY,
    }
    try:
        find_res = requests.get(find_url, params=find_params, timeout=15)
        find_json = find_res.json()
    except Exception as e:
        return BusinessResponse(found=False, error=f"FindPlace error: {e}")

    if find_json.get("status") != "OK" or not find_json.get("candidates"):
        return BusinessResponse(found=False, error=find_json.get("status", "Not Found"))

    candidate = find_json["candidates"][0]
    place_id = candidate["place_id"]

    # 2) Place Details
    details_url = "https://maps.googleapis.com/maps/api/place/details/json"
    details_params = {
        "place_id": place_id,
        "fields": "place_id,name,formatted_address,website,url,rating,user_ratings_total,opening_hours,types",
        "key": PLACES_API_KEY,
    }
    try:
        details_res = requests.get(details_url, params=details_params, timeout=15)
        details_json = details_res.json()
    except Exception as e:
        return BusinessResponse(found=False, error=f"Place Details error: {e}")

    if details_json.get("status") != "OK":
        return BusinessResponse(found=False, error=details_json.get("status", "DetailsError"))

    r = details_json["result"]

    return BusinessResponse(
        found=True,
        place_id=r.get("place_id"),
        name=r.get("name"),
        formatted_address=r.get("formatted_address"),
        rating=r.get("rating"),
        user_ratings_total=r.get("user_ratings_total"),
        website=r.get("website"),
        google_maps_url=r.get("url"),
        categories=r.get("types"),
        open_now=(r.get("opening_hours", {}) or {}).get("open_now"),
    )
