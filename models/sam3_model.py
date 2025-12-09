"""
SAM3 Model Wrapper for Interactive Segmentation
Supports point (click) and box prompts for mask generation
"""
import os
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
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


class SAM3Model:
    """
    SAM3 wrapper for interactive segmentation.
    Supports point and box prompts.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or get_device()
        self.model = None
        self.processor = None
        self.current_image = None
        self.image_set = False
        
        logger.info(f"Using device: {self.device}")
        self._load_model(model_path)
    
    def _load_model(self, model_path: Optional[str] = None):
        """Load SAM3 model"""
        try:
            import torch
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            logger.info("Loading SAM3 model...")
            
            # Build SAM3 model
            self.model = build_sam3_image_model()
            
            # Move to device
            if self.device == "cuda":
                self.model = self.model.cuda()
            elif self.device == "mps":
                self.model = self.model.to("mps")
            
            self.processor = Sam3Processor(self.model)
            
            logger.info(f"SAM3 model loaded successfully on {self.device}")
        except ImportError as e:
            logger.error(f"SAM3 not installed. Install with: pip install git+https://github.com/facebookresearch/sam3.git")
            raise ImportError(
                "SAM3 is not installed. Please install it with:\n"
                "pip install git+https://github.com/facebookresearch/sam3.git"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load SAM3 model: {e}")
            raise
    
    def set_image(self, image: Image.Image) -> bool:
        """
        Set the image for segmentation.
        This computes the image embeddings.
        """
        try:
            self.current_image = image
            
            # Convert PIL to numpy if needed
            image_np = np.array(image)
            self.processor.set_image(image_np)
            self.image_set = True
            logger.info(f"Image set: {image.size}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to set image: {e}")
            return False
    
    def predict_mask(
        self,
        point_coords: Optional[List[Tuple[int, int]]] = None,
        point_labels: Optional[List[int]] = None,
        box: Optional[Tuple[int, int, int, int]] = None,
        multimask_output: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict mask(s) from prompts.
        
        Args:
            point_coords: List of (x, y) coordinates for point prompts
            point_labels: List of labels (1 for foreground, 0 for background)
            box: Bounding box as (x1, y1, x2, y2)
            multimask_output: Whether to return multiple masks
            
        Returns:
            masks: Binary masks [N, H, W]
            scores: Confidence scores [N]
            logits: Raw logits [N, H, W]
        """
        if not self.image_set:
            raise ValueError("Image not set. Call set_image() first.")
        
        # Prepare inputs
        input_point = None
        input_label = None
        input_box = None
        
        if point_coords is not None:
            input_point = np.array(point_coords)
            input_label = np.array(point_labels if point_labels else [1] * len(point_coords))
        
        if box is not None:
            input_box = np.array(box)
        
        # Run prediction
        masks, scores, logits = self.processor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=multimask_output
        )
        
        return masks, scores, logits
    
    def reset(self):
        """Reset the model state"""
        self.current_image = None
        self.image_set = False


# Singleton instance
_sam3_model: Optional[SAM3Model] = None


def get_sam3_model() -> SAM3Model:
    """Get or create SAM3 model instance"""
    global _sam3_model
    if _sam3_model is None:
        _sam3_model = SAM3Model()
    return _sam3_model
