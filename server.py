import os
import re

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn


# Import your router
from api.routes import router as ai_router
from api.finance_routes import finance_router

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
        "https://multi-modal-agent-frontend.vercel.app",
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
_origin_pattern = re.compile(origin_regex) if origin_regex else None


def _origin_allowed(origin: str | None) -> bool:
    if not origin:
        return False

    normalized = origin.strip().rstrip("/")
    if normalized in origins:
        return True

    if _origin_pattern and _origin_pattern.match(normalized):
        return True

    return False

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def ensure_cors_headers(request, call_next):
    """
    Defensive CORS fallback & fast-path for preflight.
    """
    origin = request.headers.get("origin")

    # Intercept OPTIONS fully to ensure corporate proxies and strict modes pass
    if request.method == "OPTIONS":
        response = Response(status_code=204)
        if _origin_allowed(origin):
            response.headers["Access-Control-Allow-Origin"] = origin.strip().rstrip("/")
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,PATCH,DELETE,OPTIONS"
            req_headers = request.headers.get("access-control-request-headers", "*")
            response.headers["Access-Control-Allow-Headers"] = req_headers
            response.headers["Vary"] = "Origin"
        return response

    try:
        response = await call_next(request)
    except Exception as exc:
        response = JSONResponse(status_code=500, content={"detail": f"Internal Server Error: {str(exc)}"})

    if _origin_allowed(origin) and "access-control-allow-origin" not in response.headers:
        response.headers["Access-Control-Allow-Origin"] = origin.strip().rstrip("/")
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Vary"] = "Origin"

        req_headers = request.headers.get("access-control-request-headers", "*")
        if req_headers:
            response.headers["Access-Control-Allow-Headers"] = req_headers
        if "access-control-allow-methods" not in response.headers:
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,PATCH,DELETE,OPTIONS"

    return response

# ---------------------------------------
# Include AI Router
# ---------------------------------------

app.include_router(ai_router)
app.include_router(finance_router)

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
