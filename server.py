from fastapi import FastAPI
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

origins = [
    "http://localhost:5173",     # React dev server
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "https://z8g765mn-8000.inc1.devtunnels.ms",
    "https://z8g765mn-5173.inc1.devtunnels.ms",
    # Add your production frontend URL later
    # "https://yourdomain.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # Or ["*"] during development
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
