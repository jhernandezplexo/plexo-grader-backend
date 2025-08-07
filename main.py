from fastapi import FastAPI
from fastapi import Query
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
def scan(url: str = Query(...), email: str = Query(...)):
    zapier_webhook_url = "https://hooks.zapier.com/hooks/catch/9116429/u6ozr8x/"  # Replace this
    zapier_payload = {
        "email": email,
        "url": url
    }
    try:
        requests.post(zapier_webhook_url, json=zapier_payload)
    except Exception as e:
        print("Error posting to Zapier:", e)