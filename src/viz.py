# Visualization utilities
import numpy as np
from pathlib import Path
from PIL import Image
import cv2


def normalize_map(anomaly_map):
    """
    Normalize anomaly map to [0, 1] range.
    
    Args:
        anomaly_map: numpy array or PIL Image, can be any shape
        
    Returns:
        numpy array in [0, 1] range
    """
    # Convert PIL Image to numpy if needed
    if isinstance(anomaly_map, Image.Image):
        anomaly_map = np.array(anomaly_map)
    
    # Ensure it's a numpy array
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    
    # Normalize to [0, 1]
    min_val = anomaly_map.min()
    max_val = anomaly_map.max()
    
    if max_val > min_val:
        normalized = (anomaly_map - min_val) / (max_val - min_val)
    else:
        # All values are the same
        normalized = np.zeros_like(anomaly_map)
    
    return normalized


def to_heatmap(anomaly_map_norm):
    """
    Convert normalized anomaly map to uint8 heatmap using colormap.
    
    Args:
        anomaly_map_norm: numpy array in [0, 1] range
        
    Returns:
        uint8 numpy array (H, W, 3) RGB heatmap
    """
    # Ensure it's numpy and in [0, 1] range
    anomaly_map_norm = np.asarray(anomaly_map_norm, dtype=np.float32)
    anomaly_map_norm = np.clip(anomaly_map_norm, 0.0, 1.0)
    
    # Convert to uint8 [0, 255] for colormap
    anomaly_map_uint8 = (anomaly_map_norm * 255).astype(np.uint8)
    
    # Apply colormap (JET colormap: blue -> green -> red)
    heatmap = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)
    
    # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return heatmap


def overlay_heatmap(rgb_image, heatmap, alpha=0.5):
    """
    Overlay heatmap on RGB image.
    
    Args:
        rgb_image: numpy array (H, W, 3) or PIL Image, RGB format
        heatmap: numpy array (H, W, 3) RGB heatmap
        alpha: float, blending factor (0.0 = only image, 1.0 = only heatmap)
        
    Returns:
        numpy array (H, W, 3) RGB overlay image
    """
    # Convert PIL Image to numpy if needed
    if isinstance(rgb_image, Image.Image):
        rgb_image = np.array(rgb_image)
    
    # Ensure both are numpy arrays
    rgb_image = np.asarray(rgb_image, dtype=np.float32)
    heatmap = np.asarray(heatmap, dtype=np.float32)
    
    # Ensure same shape
    if rgb_image.shape[:2] != heatmap.shape[:2]:
        # Resize heatmap to match image
        h, w = rgb_image.shape[:2]
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1] if needed
    if rgb_image.max() > 1.0:
        rgb_image = rgb_image / 255.0
    if heatmap.max() > 1.0:
        heatmap = heatmap / 255.0
    
    # Blend: overlay = (1 - alpha) * image + alpha * heatmap
    overlay = (1 - alpha) * rgb_image + alpha * heatmap
    
    # Convert back to uint8 [0, 255]
    overlay = (overlay * 255).astype(np.uint8)
    
    return overlay


def save_triplet(out_dir, base_name, input_img, anomaly_map):
    """
    Save triplet: input image, heatmap, and overlay.
    
    Args:
        out_dir: Path or str, output directory
        base_name: str, base filename (without extension)
        input_img: numpy array or PIL Image, input RGB image
        anomaly_map: numpy array or PIL Image, anomaly map
    """
    # Create output directory
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert PIL Images to numpy if needed
    if isinstance(input_img, Image.Image):
        input_img = np.array(input_img)
    if isinstance(anomaly_map, Image.Image):
        anomaly_map = np.array(anomaly_map)
    
    # Ensure numpy arrays
    input_img = np.asarray(input_img)
    anomaly_map = np.asarray(anomaly_map)
    
    # Normalize anomaly map
    anomaly_map_norm = normalize_map(anomaly_map)
    
    # Generate heatmap
    heatmap = to_heatmap(anomaly_map_norm)
    
    # Create overlay
    overlay = overlay_heatmap(input_img, heatmap, alpha=0.5)
    
    # Save input image
    input_path = out_dir / f"{base_name}_input.png"
    if input_img.dtype != np.uint8:
        input_img_save = (np.clip(input_img, 0, 255)).astype(np.uint8)
    else:
        input_img_save = input_img
    Image.fromarray(input_img_save).save(input_path)
    
    # Save heatmap
    heatmap_path = out_dir / f"{base_name}_heatmap.png"
    Image.fromarray(heatmap).save(heatmap_path)
    
    # Save overlay
    overlay_path = out_dir / f"{base_name}_overlay.png"
    Image.fromarray(overlay).save(overlay_path)
    
    print(f"Saved triplet to {out_dir}:")
    print(f"  - {base_name}_input.png")
    print(f"  - {base_name}_heatmap.png")
    print(f"  - {base_name}_overlay.png")

