import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional, List, Union
import logging
from pathlib import Path

from rubber_sheet import unwrap_rubber_sheet
from RITnet.densenet import DenseNet2D
from RITnet.dataset import transform as rit_transform

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global segmentation model - loaded once for efficiency
_seg_model = None
_device = None

def get_segmentation_model():
    """Get or initialize the global segmentation model."""
    global _seg_model, _device
    if _seg_model is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _seg_model = DenseNet2D(dropout=True, prob=0.2).to(_device)
        

        model_dir  = Path(__file__).parents[0] / 'RITnet'
        model_path = model_dir / 'best_model.pkl' 
        
        _seg_model.load_state_dict(torch.load(model_path, map_location=_device))
        _seg_model.eval()
        logger.info(f"Loaded RITnet segmentation model on {_device}")
    
    return _seg_model, _device

def segment_iris(img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Run RITnet segmentation to get pupil/iris circles and build binary annulus mask.
    
    Args:
        img: Grayscale image (H, W)
    
    Returns:
        tuple: (annulus_mask, pupil_circle, iris_circle)
            - annulus_mask: Binary mask where 1=iris region, 0=background
            - pupil_circle: (x, y, radius) of pupil
            - iris_circle: (x, y, radius) of iris
    """
    seg_model, device = get_segmentation_model()
    
    # Preprocess for RITnet
    inp = rit_transform(img)                    # (1, H, W) tensor
    inp = inp.unsqueeze(0).to(device)           # (1, 1, H, W)
    
    # Run segmentation
    with torch.no_grad():
        out = seg_model(inp)                    # (1, 4, H, W)
    
    probs = out.squeeze(0).cpu().numpy()        # (4, H, W)
    labels = np.argmax(probs, axis=0).astype(np.uint8)  # (H, W)
    
    # Extract pupil (label=3)
    pupil_mask = (labels == 3).astype(np.uint8) * 255
    pupil_contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not pupil_contours:
        # Fallback: assume center pupil
        h, w = img.shape
        pupil_circle = (w//2, h//2, min(w, h)//8)
        logger.warning("No pupil contours found, using fallback")
    else:
        largest_pupil = max(pupil_contours, key=cv2.contourArea)
        (x_p, y_p), r_p = cv2.minEnclosingCircle(largest_pupil)
        pupil_circle = (float(x_p), float(y_p), float(r_p))
    
    # Extract iris (label=2)
    iris_mask = (labels == 2).astype(np.uint8) * 255
    iris_contours, _ = cv2.findContours(iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not iris_contours:
        # Fallback: assume iris is 3x pupil radius
        h, w = img.shape
        iris_circle = (pupil_circle[0], pupil_circle[1], pupil_circle[2] * 3)
        logger.warning("No iris contours found, using fallback")
    else:
        largest_iris = max(iris_contours, key=cv2.contourArea)
        (x_i, y_i), r_i = cv2.minEnclosingCircle(largest_iris)
        iris_circle = (float(x_i), float(y_i), float(r_i))
    
    # Create annulus mask (iris region with pupil hole)
    annulus_mask = np.zeros_like(labels, dtype=np.uint8)
    
    # Fill iris circle
    cv2.circle(annulus_mask, 
               (int(iris_circle[0]), int(iris_circle[1])), 
               int(iris_circle[2]), 1, -1)
    
    # Remove pupil circle
    cv2.circle(annulus_mask, 
               (int(pupil_circle[0]), int(pupil_circle[1])), 
               int(pupil_circle[2]), 0, -1)
    
    return annulus_mask, pupil_circle, iris_circle

def segment_and_unwrap(img: np.ndarray, output_shape: Tuple[int, int] = (64, 512)) -> np.ndarray:
    """
    Complete pipeline: segment iris → unwrap to polar coordinates.
    
    Args:
        img: Grayscale input image (H, W)
        output_shape: (num_radii, num_angles) for polar output
    
    Returns:
        np.ndarray: Polar-unwrapped iris image of shape output_shape
    """
    try:
        mask, pupil_circle, iris_circle = segment_iris(img)
        polar_img = unwrap_rubber_sheet(img, mask, pupil_circle, iris_circle, output_shape)
        return polar_img
    except Exception as e:
        logger.error(f"Failed to segment and unwrap image: {e}")
        # Return zeros as fallback
        return np.zeros(output_shape, dtype=np.float32)

def parse_subject_id(img_path: str) -> int:
    """Extract subject ID from image path."""
    parts = Path(img_path).parts
    # Assuming format: dataset/DATASET_NAME/SUBJECT_ID/...
    if len(parts) >= 3:
        try:
            return int(parts[2])  
        except ValueError:
            pass
    
    # Fallback: try to find numeric part in filename
    filename = Path(img_path).stem
    numeric_parts = ''.join(filter(str.isdigit, filename))
    if numeric_parts:
        return int(numeric_parts[:3])  # Take first 3 digits
    
    logger.warning(f"Could not parse subject ID from {img_path}, using 0")
    return 0
def normalize_polar(x: torch.Tensor) -> torch.Tensor:
    """Normalize polar tensor to [-1, 1] range."""
    return (x - 0.5) / 0.5

def create_transforms():
    """Create standard transforms for polar iris images."""
    return normalize_polar

# ————————————————————————————————————————————————————————————————————————————————————————
# Dataset Classes
# ————————————————————————————————————————————————————————————————————————————————————————