"""
Segmentation API Router
Handles image upload and mask generation using SAM3
"""
import os
import uuid
import logging
from typing import List, Optional
from PIL import Image
import numpy as np
import io
import base64

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from models.sam3_model import get_sam3_model

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/segment", tags=["segmentation"])

# Store for current session data
session_data = {}


class PointPrompt(BaseModel):
    x: int
    y: int
    label: int = 1  # 1 for foreground, 0 for background


class SegmentRequest(BaseModel):
    session_id: str
    points: List[PointPrompt]
    box: Optional[List[int]] = None  # [x1, y1, x2, y2]


class SegmentResponse(BaseModel):
    success: bool
    session_id: str
    masks: List[str]  # Base64 encoded masks
    scores: List[float]
    selected_mask_index: int = 0


def mask_to_base64(mask: np.ndarray) -> str:
    """Convert mask array to base64 PNG"""
    # Ensure mask is 2D and uint8
    if mask.ndim == 3:
        mask = mask[0]
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Convert to PIL Image
    mask_img = Image.fromarray(mask_uint8, mode='L')
    
    # Save to bytes
    buffer = io.BytesIO()
    mask_img.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def create_colored_mask(mask: np.ndarray, color: tuple = (0, 120, 255)) -> str:
    """Create colored mask overlay as base64 PNG with transparency"""
    if mask.ndim == 3:
        mask = mask[0]
    
    h, w = mask.shape
    colored = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Set color where mask is True
    mask_bool = mask > 0
    colored[mask_bool, 0] = color[0]  # R
    colored[mask_bool, 1] = color[1]  # G
    colored[mask_bool, 2] = color[2]  # B
    colored[mask_bool, 3] = 128  # Alpha (semi-transparent)
    
    # Convert to PIL Image
    mask_img = Image.fromarray(colored, mode='RGBA')
    
    # Save to bytes
    buffer = io.BytesIO()
    mask_img.save(buffer, format='PNG')
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image and get a session ID.
    The image will be processed and ready for segmentation.
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Save original image
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        image_path = os.path.join(upload_dir, f"{session_id}.png")
        image.save(image_path)
        
        # Set image in SAM3 model
        sam3 = get_sam3_model()
        sam3.set_image(image)
        
        # Store session data
        session_data[session_id] = {
            "image_path": image_path,
            "image_size": image.size,
            "masks": None,
            "scores": None,
            "selected_mask": None
        }
        
        # Create thumbnail for response
        thumbnail = image.copy()
        thumbnail.thumbnail((800, 800))
        buffer = io.BytesIO()
        thumbnail.save(buffer, format='PNG')
        buffer.seek(0)
        thumbnail_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        logger.info(f"Image uploaded: {session_id}, size: {image.size}")
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "image_size": {"width": image.size[0], "height": image.size[1]},
            "thumbnail": thumbnail_b64
        })
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict_mask(
    session_id: str = Form(...),
    points_x: str = Form(...),  # Comma-separated x coordinates
    points_y: str = Form(...),  # Comma-separated y coordinates
    labels: str = Form(...)     # Comma-separated labels (1=fg, 0=bg)
):
    """
    Predict mask from point prompts.
    """
    try:
        # Validate session
        if session_id not in session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Parse points
        x_coords = [int(x) for x in points_x.split(',') if x]
        y_coords = [int(y) for y in points_y.split(',') if y]
        point_labels = [int(l) for l in labels.split(',') if l]
        
        if len(x_coords) != len(y_coords) or len(x_coords) != len(point_labels):
            raise HTTPException(status_code=400, detail="Point coordinates and labels must match")
        
        point_coords = list(zip(x_coords, y_coords))
        
        # Load image and set in model
        image_path = session_data[session_id]["image_path"]
        image = Image.open(image_path)
        
        sam3 = get_sam3_model()
        sam3.set_image(image)
        
        # Predict masks
        masks, scores, logits = sam3.predict_mask(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        
        # Store masks in session
        session_data[session_id]["masks"] = masks
        session_data[session_id]["scores"] = scores
        session_data[session_id]["selected_mask"] = masks[0]  # Default to first mask
        
        # Convert masks to base64 colored overlays
        colors = [
            (0, 120, 255),   # Blue
            (0, 255, 120),   # Green  
            (255, 120, 0),   # Orange
        ]
        
        mask_images = []
        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            mask_b64 = create_colored_mask(mask, color)
            mask_images.append(mask_b64)
        
        logger.info(f"Masks predicted: {len(masks)} masks, scores: {scores}")
        
        return JSONResponse({
            "success": True,
            "session_id": session_id,
            "masks": mask_images,
            "scores": scores.tolist(),
            "selected_mask_index": 0,
            "num_masks": len(masks)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select-mask")
async def select_mask(session_id: str = Form(...), mask_index: int = Form(...)):
    """
    Select a specific mask from the predictions.
    """
    try:
        if session_id not in session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        masks = session_data[session_id].get("masks")
        if masks is None:
            raise HTTPException(status_code=400, detail="No masks predicted yet")
        
        if mask_index < 0 or mask_index >= len(masks):
            raise HTTPException(status_code=400, detail="Invalid mask index")
        
        session_data[session_id]["selected_mask"] = masks[mask_index]
        
        return JSONResponse({
            "success": True,
            "selected_mask_index": mask_index
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Mask selection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    if session_id not in session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = session_data[session_id]
    return JSONResponse({
        "session_id": session_id,
        "image_size": session.get("image_size"),
        "has_masks": session.get("masks") is not None,
        "num_masks": len(session.get("masks", [])) if session.get("masks") is not None else 0
    })


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its data"""
    if session_id in session_data:
        # Delete image file
        image_path = session_data[session_id].get("image_path")
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        
        del session_data[session_id]
    
    return JSONResponse({"success": True})

