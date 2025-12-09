"""
3D Extractor Server
SAM3 + SAM 3D Objects based 3D extraction from images

Usage:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import os
import logging
from contextlib import asynccontextmanager

import torch
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


def get_device_info() -> dict:
    """Get device information"""
    if torch.cuda.is_available():
        return {
            "type": "cuda",
            "name": torch.cuda.get_device_name(0),
            "memory": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        }
    elif torch.backends.mps.is_available():
        return {
            "type": "mps",
            "name": "Apple Metal (MPS)",
            "memory": "Shared"
        }
    else:
        return {
            "type": "cpu",
            "name": "CPU",
            "memory": "System RAM"
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("Starting 3D Extractor Server...")
    
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs/masks", exist_ok=True)
    os.makedirs("outputs/gaussians", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    
    # Log device info
    device_info = get_device_info()
    logger.info(f"Device: {device_info['type']} - {device_info['name']}")
    
    # Models will be loaded lazily on first request
    logger.info("Models will be loaded on first request")
    
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
    device_info = get_device_info()
    
    return {
        "service": "3D Extractor",
        "version": "1.0.0",
        "device": device_info,
        "models": {
            "sam3": {
                "status": "ready"
            },
            "sam3d": {
                "status": "ready"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
