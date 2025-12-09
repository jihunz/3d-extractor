"""
SAM 3D Objects Model Wrapper for 3D Reconstruction
Converts masked images to 3D Gaussian Splatting
"""
import os
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_device() -> str:
    """Get the best available device (CUDA > MPS > CPU)"""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class SAM3DModel:
    """
    SAM 3D Objects wrapper for 3D reconstruction.
    Converts masked images to Gaussian Splatting PLY files.
    """
    
    def __init__(self, config_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or get_device()
        self.inference = None
        self.config_path = config_path
        
        logger.info(f"Using device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load SAM 3D Objects model"""
        try:
            import torch
            from sam3d_objects.inference import Inference
            
            logger.info("Loading SAM 3D Objects model...")
            
            # Default config path
            if self.config_path is None:
                self.config_path = "checkpoints/hf/pipeline.yaml"
            
            self.inference = Inference(self.config_path, compile=False, device=self.device)
            
            logger.info(f"SAM 3D Objects model loaded successfully on {self.device}")
        except ImportError as e:
            logger.error(f"SAM 3D Objects not installed. Install with: pip install git+https://github.com/facebookresearch/sam-3d-objects.git")
            raise ImportError(
                "SAM 3D Objects is not installed. Please install it with:\n"
                "pip install git+https://github.com/facebookresearch/sam-3d-objects.git"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load SAM 3D Objects model: {e}")
            raise
    
    def reconstruct_3d(
        self,
        image: Image.Image,
        mask: np.ndarray,
        seed: int = 42,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Reconstruct 3D from masked image.
        
        Args:
            image: PIL Image
            mask: Binary mask (H, W) or (N, H, W)
            seed: Random seed for reproducibility
            output_path: Path to save PLY file
            
        Returns:
            Dictionary with:
                - success: bool
                - ply_path: Path to saved PLY file
                - mesh_data: Additional mesh data if available
        """
        try:
            from sam3d_objects.inference import (
                ready_gaussian_for_video_rendering,
                make_scene
            )
            
            # Convert PIL to numpy
            image_np = np.array(image)
            
            # Ensure mask is 2D
            if mask.ndim == 3:
                mask = mask[0]  # Take first mask
            
            # Run inference
            output = self.inference(image_np, mask, seed=seed)
            
            # Save PLY file
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                output["gs"].save_ply(output_path)
                logger.info(f"Saved PLY to: {output_path}")
            
            # Prepare scene for rendering
            scene_gs = make_scene(output)
            scene_gs = ready_gaussian_for_video_rendering(scene_gs)
            
            return {
                "success": True,
                "ply_path": output_path,
                "gaussian_data": output.get("gs"),
                "mesh_data": output.get("mesh")
            }
        except Exception as e:
            logger.error(f"3D reconstruction failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Singleton instance
_sam3d_model: Optional[SAM3DModel] = None


def get_sam3d_model() -> SAM3DModel:
    """Get or create SAM 3D Objects model instance"""
    global _sam3d_model
    if _sam3d_model is None:
        _sam3d_model = SAM3DModel()
    return _sam3d_model
