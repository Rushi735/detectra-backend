from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.api.endpoints import analysis, auth
from app.db.session import engine, Base
from app.models.user import User 
from app.models.analysis_log import AnalysisLog # Register AnalysisLog

# Create tables
Base.metadata.create_all(bind=engine)

settings = get_settings()

app = FastAPI(title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(analysis.router, prefix=f"{settings.API_V1_STR}/analysis", tags=["analysis"])

@app.get("/")
def root():
    return {"message": "Welcome to Detectra API"}
