from fastapi import FastAPI
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
from pydantic import BaseModel
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials

app = FastAPI()

# Allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.vercel.app", "http://localhost:3000"],  # You can replace * with your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend is live!"}
@app.get("/scan")
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
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict({
        "type": "service_account",
        "project_id": os.getenv("GOOGLE_PROJECT_ID"),
        "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
        "private_key": os.getenv("GOOGLE_PRIVATE_KEY").replace('\\n', '\n'),
        "client_email": os.getenv("GOOGLE_SERVICE_ACCOUNT_EMAIL"),
        "client_id": os.getenv("GOOGLE_CLIENT_ID"),
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_X509_URL"),
    }, scope)

    client = gspread.authorize(creds)
    sheet = client.open_by_key(os.getenv("GOOGLE_SHEET_ID")).sheet1
    sheet.append_row([
        datetime.utcnow().isoformat(),
        lead.name,
        lead.email,
        lead.business_name,
        lead.website
    ])

def trigger_email_automation(lead: Lead):
    requests.post(os.getenv("N8N_WEBHOOK_URL"), json=lead.dict())
