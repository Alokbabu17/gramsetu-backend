from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router, get_admin_dashboard

app = FastAPI(title="GramSetu API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root level /admin endpoint (bina /api ke kholne ke liye)
app.add_api_route("/admin", get_admin_dashboard, methods=["GET"])

# Baaki saare routes /api ke sath
app.include_router(api_router, prefix="/api")