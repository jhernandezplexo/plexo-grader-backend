from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

# Allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace * with your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Backend is live!"}
@app.get("/scan")
def scan(url: str):
    api_key = os.getenv("PSI_API_KEY")
    psi_url = f"https://www.googleapis.com/pagespeedonline/v5/runPagespeed?url={url}&key={api_key}"
    
    response = requests.get(psi_url)
    if response.status_code == 200:
        data = response.json()
        score = data["lighthouseResult"]["categories"]["performance"]["score"] * 100
        return {
            "url": url,
            "performance_score": score
        }

    return {"error": "Failed to fetch PSI data"}
