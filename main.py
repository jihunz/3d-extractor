"""
3D Extractor Server
SAM3 + SAM 3D Objects based 3D extraction from images

Usage:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import routers
from routers import segment, reconstruct


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("Starting 3D Extractor Server...")
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs/masks", exist_ok=True)
    os.makedirs("outputs/gaussians", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Pre-load models (optional, can be lazy loaded)
    # from models.sam3_model import get_sam3_model
    # from models.sam3d_model import get_sam3d_model
    # get_sam3_model()
    # get_sam3d_model()
    
    logger.info("Server started successfully!")
    yield
    logger.info("Shutting down server...")


# Create FastAPI app
app = FastAPI(
    title="3D Extractor",
    description="Extract 3D Gaussian Splatting from images using SAM3 + SAM 3D Objects",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(segment.router)
app.include_router(reconstruct.router)


@app.get("/")
async def root():
    """Serve the main page"""
    return FileResponse("static/index.html")


@app.get("/viewer")
async def viewer():
    """Serve the 3D viewer page"""
    return FileResponse("static/viewer.html")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "3D Extractor",
        "version": "1.0.0"
    }


@app.get("/api/info")
async def get_info():
    """Get server information and model status"""
    from models.sam3_model import SAM3_AVAILABLE
    from models.sam3d_model import SAM3D_AVAILABLE
    
    return {
        "service": "3D Extractor",
        "version": "1.0.0",
        "models": {
            "sam3": {
                "available": SAM3_AVAILABLE,
                "status": "loaded" if SAM3_AVAILABLE else "mock_mode"
            },
            "sam3d": {
                "available": SAM3D_AVAILABLE,
                "status": "loaded" if SAM3D_AVAILABLE else "mock_mode"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
