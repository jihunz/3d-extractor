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

# Try to import SAM 3D Objects, fall back to mock mode if not available
try:
    import torch
    # SAM 3D Objects imports
    from sam3d_objects.inference import Inference
    from sam3d_objects.inference import (
        ready_gaussian_for_video_rendering,
        make_scene,
        load_image
    )
    SAM3D_AVAILABLE = True
except ImportError:
    SAM3D_AVAILABLE = False
    logger.warning("SAM 3D Objects not installed. Running in mock mode.")


class SAM3DModel:
    """
    SAM 3D Objects wrapper for 3D reconstruction.
    Converts masked images to Gaussian Splatting PLY files.
    """
    
    def __init__(self, config_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.inference = None
        self.config_path = config_path
        
        if SAM3D_AVAILABLE:
            self._load_model()
        else:
            logger.info("Running SAM 3D Objects in mock mode")
    
    def _load_model(self):
        """Load SAM 3D Objects model"""
        try:
            logger.info("Loading SAM 3D Objects model...")
            
            # Default config path
            if self.config_path is None:
                self.config_path = "checkpoints/hf/pipeline.yaml"
            
            self.inference = Inference(self.config_path, compile=False)
            
            logger.info("SAM 3D Objects model loaded successfully")
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
        if SAM3D_AVAILABLE and self.inference:
            return self._reconstruct_real(image, mask, seed, output_path)
        else:
            return self._reconstruct_mock(image, mask, seed, output_path)
    
    def _reconstruct_real(
        self,
        image: Image.Image,
        mask: np.ndarray,
        seed: int,
        output_path: Optional[str]
    ) -> Dict[str, Any]:
        """Real SAM 3D Objects reconstruction"""
        try:
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
    
    def _reconstruct_mock(
        self,
        image: Image.Image,
        mask: np.ndarray,
        seed: int,
        output_path: Optional[str]
    ) -> Dict[str, Any]:
        """Mock reconstruction for testing without SAM 3D Objects"""
        try:
            # Generate mock PLY data
            ply_data = self._generate_mock_ply(image, mask)
            
            # Save PLY file
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'wb') as f:
                    f.write(ply_data)
                logger.info(f"Saved mock PLY to: {output_path}")
            
            return {
                "success": True,
                "ply_path": output_path,
                "mock": True,
                "message": "Mock PLY generated (SAM 3D Objects not installed)"
            }
        except Exception as e:
            logger.error(f"Mock reconstruction failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_mock_ply(self, image: Image.Image, mask: np.ndarray) -> bytes:
        """Generate a mock PLY file for testing"""
        # Get mask points
        if mask.ndim == 3:
            mask = mask[0]
        
        # Find mask coordinates
        y_coords, x_coords = np.where(mask > 0)
        
        if len(x_coords) == 0:
            # No mask, create a simple cube
            points = [
                (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)
            ]
            colors = [(255, 0, 0)] * 8
        else:
            # Sample points from mask
            num_points = min(1000, len(x_coords))
            indices = np.random.choice(len(x_coords), num_points, replace=False)
            
            # Convert to normalized 3D coordinates
            img_array = np.array(image)
            h, w = mask.shape
            
            points = []
            colors = []
            for idx in indices:
                x, y = x_coords[idx], y_coords[idx]
                # Normalize to [-1, 1]
                nx = (x / w) * 2 - 1
                ny = (y / h) * 2 - 1
                nz = np.random.uniform(-0.1, 0.1)  # Small z variation
                points.append((nx, ny, nz))
                
                # Get color from image
                if y < img_array.shape[0] and x < img_array.shape[1]:
                    color = img_array[y, x]
                    if len(color) >= 3:
                        colors.append((color[0], color[1], color[2]))
                    else:
                        colors.append((128, 128, 128))
                else:
                    colors.append((128, 128, 128))
        
        # Generate PLY content
        header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        
        body = ""
        for (x, y, z), (r, g, b) in zip(points, colors):
            body += f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n"
        
        return (header + body).encode('utf-8')


# Singleton instance
_sam3d_model: Optional[SAM3DModel] = None


def get_sam3d_model() -> SAM3DModel:
    """Get or create SAM 3D Objects model instance"""
    global _sam3d_model
    if _sam3d_model is None:
        _sam3d_model = SAM3DModel()
    return _sam3d_model

