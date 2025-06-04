import numpy as np
import cv2
from typing import Tuple

def unwrap_rubber_sheet(
    image: np.ndarray,
    mask: np.ndarray,
    inner_circle: Tuple[int, int, float],
    outer_circle: Tuple[int, int, float],
    shape: tuple[int, int] = (64, 512)
) -> np.ndarray:
    """
    Daugman rubber-sheet normalization on a masked iris image.
    
    This function performs polar transformation of the iris annular region,
    converting from Cartesian coordinates to polar coordinates. The iris
    region between the pupil and iris boundaries is unwrapped into a
    rectangular representation.
    
    Args:
        image: (H,W) grayscale image, typically uint8 or float32
        mask: (H,W) binary mask where iris region = 1, background = 0
        inner_circle: (x_center, y_center, radius) for pupil boundary
        outer_circle: (x_center, y_center, radius) for iris boundary
        shape: (num_radii, num_angles) output polar image dimensions
        
    Returns:
        (num_radii, num_angles) normalized float32 array in [0,1]
    """
    # Ensure inputs are proper types
    if image.dtype == np.uint8:
        img = image.astype(np.float32) / 255.0
    else:
        img = image.astype(np.float32)
    
    # Apply mask to isolate iris region
    masked_img = img * mask.astype(np.float32)
    
    H, W = img.shape
    x_in, y_in, r_in = inner_circle
    x_out, y_out, r_out = outer_circle
    num_r, num_theta = shape
    
    # Handle potential floating point coordinates
    x_in, y_in, r_in = float(x_in), float(y_in), float(r_in)
    x_out, y_out, r_out = float(x_out), float(y_out), float(r_out)
    
    # Create radial and angular sampling arrays
    radii = np.linspace(r_in, r_out, num_r)
    thetas = np.linspace(0, 2*np.pi, num_theta, endpoint=False)
    
    # Initialize output polar image
    polar = np.zeros((num_r, num_theta), dtype=np.float32)
    
    # For each radius level
    for i, r in enumerate(radii):
        # Linear interpolation between inner and outer circle centers
        # This handles cases where pupil and iris centers don't align
        alpha = (r - r_in) / (r_out - r_in) if r_out != r_in else 0.0
        x_center = x_in + alpha * (x_out - x_in)
        y_center = y_in + alpha * (y_out - y_in)
        
        # Calculate Cartesian coordinates for this radius
        x_coords = x_center + r * np.cos(thetas)
        y_coords = y_center + r * np.sin(thetas)
        
        # Use bilinear interpolation for smoother results
        polar[i, :] = bilinear_interpolate(masked_img, x_coords, y_coords)
    
    return polar


def bilinear_interpolate(image: np.ndarray, x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
    """
    Perform bilinear interpolation for smoother polar transformation.
    
    Args:
        image: (H,W) source image
        x_coords: array of x coordinates to sample
        y_coords: array of y coordinates to sample
        
    Returns:
        Array of interpolated pixel values
    """
    H, W = image.shape
    
    # Clip coordinates to image bounds
    x_coords = np.clip(x_coords, 0, W - 1)
    y_coords = np.clip(y_coords, 0, H - 1)
    
    # Get integer parts and fractional parts
    x0 = np.floor(x_coords).astype(int)
    y0 = np.floor(y_coords).astype(int)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    
    # Get fractional parts
    dx = x_coords - x0
    dy = y_coords - y0
    
    # Bilinear interpolation
    top_left = image[y0, x0]
    top_right = image[y0, x1]
    bottom_left = image[y1, x0]
    bottom_right = image[y1, x1]
    
    top = top_left * (1 - dx) + top_right * dx
    bottom = bottom_left * (1 - dx) + bottom_right * dx
    result = top * (1 - dy) + bottom * dy
    
    return result


def estimate_circles_from_mask(mask: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Estimate pupil and iris circle parameters from a binary segmentation mask.
    
    This function assumes the mask has different values for pupil, iris, and background:
    - 0: background
    - 1: iris region  
    - 2: pupil region (optional, if available)
    
    Args:
        mask: (H,W) segmentation mask
        
    Returns:
        Tuple of (inner_circle, outer_circle) where each circle is (x, y, radius)
    """
    # Find iris contour (assuming iris region has value 1)
    iris_mask = (mask == 1).astype(np.uint8)
    contours, _ = cv2.findContours(iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback: use entire mask
        iris_mask = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(iris_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get largest contour (should be iris boundary)
    iris_contour = max(contours, key=cv2.contourArea)
    
    # Fit circle to iris contour
    (x_iris, y_iris), r_iris = cv2.minEnclosingCircle(iris_contour)
    
    # Try to find pupil contour
    pupil_mask = None
    if np.any(mask == 2):
        # If mask has separate pupil label
        pupil_mask = (mask == 2).astype(np.uint8)
    else:
        # Try to find dark central region as pupil
        # This is a heuristic approach
        center_region = mask.copy()
        h, w = center_region.shape
        center_y, center_x = int(y_iris), int(x_iris)
        
        # Create circular mask around estimated iris center
        y_grid, x_grid = np.ogrid[:h, :w]
        center_mask = (x_grid - center_x)**2 + (y_grid - center_y)**2 < (r_iris * 0.4)**2
        
        # Find the most common value in the center region (likely pupil)
        center_values = mask[center_mask]
        if len(center_values) > 0:
            # Use mode of center region
            unique_vals, counts = np.unique(center_values, return_counts=True)
            pupil_val = unique_vals[np.argmax(counts)]
            if pupil_val != 1:  # If it's different from iris value
                pupil_mask = (mask == pupil_val).astype(np.uint8)
    
    # Estimate pupil circle
    if pupil_mask is not None and np.any(pupil_mask):
        pupil_contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if pupil_contours:
            pupil_contour = max(pupil_contours, key=cv2.contourArea)
            (x_pupil, y_pupil), r_pupil = cv2.minEnclosingCircle(pupil_contour)
        else:
            # Fallback: assume pupil is at iris center with 1/3 radius
            x_pupil, y_pupil, r_pupil = x_iris, y_iris, r_iris * 0.3
    else:
        # Fallback: assume pupil is at iris center with 1/3 radius
        x_pupil, y_pupil, r_pupil = x_iris, y_iris, r_iris * 0.3
    
    inner_circle = (float(x_pupil), float(y_pupil), float(r_pupil))
    outer_circle = (float(x_iris), float(y_iris), float(r_iris))
    
    return inner_circle, outer_circle


def normalize_polar_image(polar_image: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize polar image for consistent preprocessing.
    
    Args:
        polar_image: (H,W) polar image array
        method: normalization method ('minmax', 'zscore', 'none')
        
    Returns:
        Normalized polar image
    """
    if method == 'minmax':
        # Min-max normalization to [0,1]
        img_min = polar_image.min()
        img_max = polar_image.max()
        if img_max > img_min:
            return (polar_image - img_min) / (img_max - img_min)
        else:
            return polar_image
    elif method == 'zscore':
        # Z-score normalization
        mean = polar_image.mean()
        std = polar_image.std()
        if std > 0:
            return (polar_image - mean) / std
        else:
            return polar_image - mean
    else:
        return polar_image


def create_three_channel_polar(polar_image: np.ndarray) -> np.ndarray:
    """
    Convert single-channel polar image to 3-channel for ConvNeXt input.
    
    Args:
        polar_image: (H,W) single channel polar image
        
    Returns:
        (3,H,W) three-channel tensor
    """
    # Simply replicate the single channel 3 times
    return np.stack([polar_image, polar_image, polar_image], axis=0)


# Additional utility function for debugging
def visualize_unwrapping(image: np.ndarray, mask: np.ndarray, 
                        inner_circle: tuple[float, float, float],
                        outer_circle: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Create visualization of the unwrapping process for debugging.
    
    Returns:
        Tuple of (annotated_original, polar_image)
    """
    # Create annotated version of original image
    vis_img = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # Draw circles
    x_in, y_in, r_in = inner_circle
    x_out, y_out, r_out = outer_circle
    
    cv2.circle(vis_img, (int(x_in), int(y_in)), int(r_in), (0, 255, 0), 2)  # Green for pupil
    cv2.circle(vis_img, (int(x_out), int(y_out)), int(r_out), (0, 0, 255), 2)  # Red for iris
    
    # Generate polar image
    polar = unwrap_rubber_sheet(image, mask, inner_circle, outer_circle)
    
    return vis_img, polar