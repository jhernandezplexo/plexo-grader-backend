from typing import Optional
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os

app = FastAPI()

# Allow requests from the frontend (add your real Vercel domain)
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


# ---------- NEW: GET /scan ----------
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


# ---------- Existing: POST /leads ----------
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


# ---------- Helpers ----------
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

