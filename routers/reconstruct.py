"""
3D Reconstruction API Router
Handles 3D reconstruction from masks using SAM 3D Objects
"""
import os
import logging
from typing import Optional
from PIL import Image
import numpy as np

from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse

from models.sam3d_model import get_sam3d_model
from routers.segment import session_data

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/reconstruct", tags=["reconstruction"])


@router.post("/generate")
async def generate_3d(
    session_id: str = Form(...),
    mask_index: int = Form(0),
    seed: int = Form(42)
):
    """
    Generate 3D Gaussian Splatting from the selected mask.
    
    Args:
        session_id: Session ID from upload
        mask_index: Index of mask to use (default: 0, best mask)
        seed: Random seed for reproducibility
        
    Returns:
        JSON with PLY file path and download URL
    """
    try:
        # Validate session
        if session_id not in session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = session_data[session_id]
        
        # Check if mask exists
        masks = session.get("masks")
        if masks is None:
            raise HTTPException(status_code=400, detail="No masks available. Run segmentation first.")
        
        if mask_index < 0 or mask_index >= len(masks):
            raise HTTPException(status_code=400, detail=f"Invalid mask index. Available: 0-{len(masks)-1}")
        
        # Get selected mask
        mask = masks[mask_index]
        
        # Load original image
        image_path = session["image_path"]
        image = Image.open(image_path)
        
        # Generate output path
        output_dir = os.path.join("outputs", "gaussians")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{session_id}.ply")
        
        # Run 3D reconstruction
        sam3d = get_sam3d_model()
        result = sam3d.reconstruct_3d(
            image=image,
            mask=mask,
            seed=seed,
            output_path=output_path
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"3D reconstruction failed: {result.get('error', 'Unknown error')}"
            )
        
        # Store result in session
        session["ply_path"] = output_path
        session["reconstruction_result"] = result
        
        logger.info(f"3D reconstruction complete: {output_path}")
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "ply_path": output_path,
            "download_url": f"/api/reconstruct/download/{session_id}",
            "mock": result.get("mock", False)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"3D reconstruction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{session_id}")
async def download_ply(session_id: str):
    """
    Download the generated PLY file.
    """
    try:
        if session_id not in session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = session_data[session_id]
        ply_path = session.get("ply_path")
        
        if not ply_path or not os.path.exists(ply_path):
            raise HTTPException(status_code=404, detail="PLY file not found. Generate 3D first.")
        
        return FileResponse(
            ply_path,
            media_type="application/octet-stream",
            filename=f"model_{session_id}.ply"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{session_id}")
async def get_status(session_id: str):
    """
    Get the status of 3D reconstruction for a session.
    """
    if session_id not in session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = session_data[session_id]
    
    return JSONResponse({
        "session_id": session_id,
        "has_image": session.get("image_path") is not None,
        "has_masks": session.get("masks") is not None,
        "has_ply": session.get("ply_path") is not None and os.path.exists(session.get("ply_path", "")),
        "ply_path": session.get("ply_path")
    })


@router.post("/batch")
async def batch_reconstruct(
    session_id: str = Form(...),
    seed: int = Form(42)
):
    """
    Generate 3D for all masks in a session.
    """
    try:
        if session_id not in session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = session_data[session_id]
        masks = session.get("masks")
        
        if masks is None:
            raise HTTPException(status_code=400, detail="No masks available")
        
        image_path = session["image_path"]
        image = Image.open(image_path)
        
        sam3d = get_sam3d_model()
        results = []
        
        output_dir = os.path.join("outputs", "gaussians", session_id)
        os.makedirs(output_dir, exist_ok=True)
        
        for i, mask in enumerate(masks):
            output_path = os.path.join(output_dir, f"mask_{i}.ply")
            result = sam3d.reconstruct_3d(
                image=image,
                mask=mask,
                seed=seed + i,
                output_path=output_path
            )
            results.append({
                "mask_index": i,
                "success": result["success"],
                "ply_path": output_path if result["success"] else None
            })
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "results": results
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch reconstruction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

