import os

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn


# Import your router
from api.routes import router as ai_router

# ---------------------------------------
# Create FastAPI App
# ---------------------------------------

app = FastAPI(
    title="MultiModal AI API",
    description="Backend API for React AI Frontend",
    version="1.0.0"
)

# ---------------------------------------
# CORS Configuration (IMPORTANT for React)
# ---------------------------------------

def _build_cors_origins() -> list[str]:
    default_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "https://z8g765mn-8000.inc1.devtunnels.ms",
        "https://z8g765mn-5173.inc1.devtunnels.ms",
        "https://multi-modal-agent-frontend-a5gy.vercel.app",
    ]

    raw_env_origins = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    if not raw_env_origins:
        return default_origins

    env_origins = [origin.strip() for origin in raw_env_origins.split(",") if origin.strip()]
    merged = default_origins + env_origins

    # Keep order stable while removing duplicates.
    unique = []
    seen = set()
    for origin in merged:
        if origin not in seen:
            seen.add(origin)
            unique.append(origin)

    return unique


origins = _build_cors_origins()
origin_regex = os.getenv("CORS_ALLOW_ORIGIN_REGEX", r"https://.*\.vercel\.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------
# Include AI Router
# ---------------------------------------

app.include_router(ai_router)

# ---------------------------------------
# Root Route
# ---------------------------------------

@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "MultiModal AI Backend is live 🚀"
    }

# ---------------------------------------
# Health Check Route
# ---------------------------------------

@app.get("/health")
async def health_check():
    return JSONResponse(
        status_code=200,
        content={"status": "healthy"}
    )


@app.head("/health", include_in_schema=False)
async def health_check_head() -> Response:
    return Response(status_code=200)

# ---------------------------------------
# Run Server
# ---------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Disable in production
    )
