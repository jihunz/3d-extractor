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

# Try to import SAM3, fall back to mock mode if not available
try:
    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    logger.warning("SAM3 not installed. Running in mock mode.")


class SAM3Model:
    """
    SAM3 wrapper for interactive segmentation.
    Supports point and box prompts.
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self.current_image = None
        self.image_set = False
        
        if SAM3_AVAILABLE:
            self._load_model(model_path)
        else:
            logger.info("Running SAM3 in mock mode")
    
    def _load_model(self, model_path: Optional[str] = None):
        """Load SAM3 model"""
        try:
            logger.info("Loading SAM3 model...")
            
            # Build SAM3 model
            self.model = build_sam3_image_model()
            self.processor = Sam3Processor(self.model)
            
            # Move to device
            if torch.cuda.is_available() and self.device == "cuda":
                self.model = self.model.cuda()
            
            logger.info("SAM3 model loaded successfully")
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
            
            if SAM3_AVAILABLE and self.processor:
                # Convert PIL to numpy if needed
                image_np = np.array(image)
                self.processor.set_image(image_np)
                self.image_set = True
                logger.info(f"Image set: {image.size}")
            else:
                self.image_set = True
                logger.info(f"Image set (mock mode): {image.size}")
            
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
        
        if SAM3_AVAILABLE and self.processor:
            return self._predict_real(point_coords, point_labels, box, multimask_output)
        else:
            return self._predict_mock(point_coords, point_labels, box, multimask_output)
    
    def _predict_real(
        self,
        point_coords: Optional[List[Tuple[int, int]]],
        point_labels: Optional[List[int]],
        box: Optional[Tuple[int, int, int, int]],
        multimask_output: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Real SAM3 prediction"""
        import torch
        
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
    
    def _predict_mock(
        self,
        point_coords: Optional[List[Tuple[int, int]]],
        point_labels: Optional[List[int]],
        box: Optional[Tuple[int, int, int, int]],
        multimask_output: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mock prediction for testing without SAM3"""
        if self.current_image is None:
            raise ValueError("No image set")
        
        h, w = self.current_image.size[1], self.current_image.size[0]
        
        # Create a mock mask based on the prompt
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if point_coords:
            # Create circular mask around the point
            for x, y in point_coords:
                radius = min(h, w) // 6
                yy, xx = np.ogrid[:h, :w]
                circle = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
                mask[circle] = 1
        
        if box:
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 1
        
        # Return mock results
        num_masks = 3 if multimask_output else 1
        masks = np.stack([mask] * num_masks)
        scores = np.array([0.95, 0.85, 0.75][:num_masks])
        logits = masks.astype(np.float32) * 10  # Mock logits
        
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

