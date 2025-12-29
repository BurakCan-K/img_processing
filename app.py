# Streamlit application for anomaly detection
import streamlit as st
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.viz import normalize_map, to_heatmap, overlay_heatmap

# Try different import paths for anomalib (API may vary between versions)
try:
    from anomalib.models import Patchcore
except ImportError:
    try:
        from anomalib.models.patchcore import Patchcore
    except ImportError:
        st.error("Could not import Patchcore from anomalib.models")
        st.stop()

# Try to import ImageBatch
ImageBatch = None
try:
    from anomalib.data.utils import ImageBatch
except ImportError:
    try:
        from anomalib.data.dataclasses.torch.image import ImageBatch
    except ImportError:
        ImageBatch = None


@st.cache_resource
def load_model(model_path):
    """Load model from checkpoint with caching."""
    try:
        model = Patchcore.load_from_checkpoint(str(model_path))
        model.eval()
        
        # Move to device
        if torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()
        
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def find_checkpoint_files(outputs_dir="./outputs"):
    """Find all checkpoint files in outputs directory."""
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        return []
    
    checkpoint_files = []
    for ckpt_path in outputs_path.rglob("*.ckpt"):
        checkpoint_files.append(ckpt_path)
    
    return sorted(checkpoint_files, reverse=True)  # Most recent first


def preprocess_image(image, target_size=256):
    """Preprocess image for model input."""
    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Resize
    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Convert to tensor and normalize (ImageNet normalization)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor, image


def predict_image(model, image_tensor):
    """Run prediction on image."""
    # Move to device
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    
    # Create batch - model expects ImageBatch format
    # Use global ImageBatch variable
    global ImageBatch
    batch = None
    
    if ImageBatch is not None:
        batch = ImageBatch(image=image_tensor)
    else:
        # Try to import ImageBatch dynamically
        try:
            from anomalib.data.utils import ImageBatch as IB
            batch = IB(image=image_tensor)
        except ImportError:
            try:
                from anomalib.data.dataclasses.torch.image import ImageBatch as IB
                batch = IB(image=image_tensor)
            except ImportError:
                # Last resort: use dict but this will likely fail
                batch = {"image": image_tensor}
    
    # Predict
    with torch.no_grad():
        try:
            output = model.test_step(batch, batch_idx=0)
        except Exception as e:
            # If test_step fails, try predict_step
            try:
                output = model.predict_step(batch, batch_idx=0)
            except Exception:
                # Last resort: use forward
                output = model(batch)
    
    # Extract results
    pred_score = None
    pred_label = None
    anomaly_map = None
    
    # Handle ImageBatch output
    if hasattr(output, "pred_score") or hasattr(output, "anomaly_score"):
        if hasattr(output, "pred_score"):
            pred_score = output.pred_score
        elif hasattr(output, "anomaly_score"):
            pred_score = output.anomaly_score
        
        if hasattr(output, "pred_label"):
            pred_label = output.pred_label
        
        if hasattr(output, "anomaly_map"):
            anomaly_map = output.anomaly_map
    elif isinstance(output, dict):
        pred_score = output.get("pred_score") or output.get("anomaly_score")
        pred_label = output.get("pred_label")
        anomaly_map = output.get("anomaly_map")
    
    # Convert tensors
    if torch.is_tensor(pred_score):
        pred_score = pred_score.cpu().item() if pred_score.numel() == 1 else pred_score.cpu().numpy()[0]
    if torch.is_tensor(pred_label):
        pred_label = int(pred_label.cpu().item() if pred_label.numel() == 1 else pred_label.cpu().numpy()[0])
    if torch.is_tensor(anomaly_map):
        anomaly_map = anomaly_map.cpu().numpy()
        if anomaly_map.ndim > 2 and anomaly_map.shape[0] == 1:
            anomaly_map = anomaly_map[0]
    
    # Infer label from score if not found
    if pred_label is None and pred_score is not None:
        pred_label = 1 if pred_score > 0.5 else 0
    
    return pred_score, pred_label, anomaly_map


def main():
    st.set_page_config(
        page_title="Anomaly Detection",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Anomaly Detection with PatchCore")
    st.markdown("Upload an image to detect anomalies using a trained PatchCore model.")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    
    # Find checkpoint files
    checkpoint_files = find_checkpoint_files()
    
    if not checkpoint_files:
        st.sidebar.warning("No checkpoint files found in `./outputs/` directory.")
        st.sidebar.info("Please train a model first using:\n```bash\npython scripts/train.py --category carpet --device cuda\n```")
        st.stop()
    
    # Model selection
    checkpoint_paths = [str(ckpt) for ckpt in checkpoint_files]
    selected_checkpoint = st.sidebar.selectbox(
        "Select Model Checkpoint",
        checkpoint_paths,
        format_func=lambda x: str(Path(x).relative_to(Path(".")))
    )
    
    # Load model
    if st.sidebar.button("Load Model", type="primary"):
        with st.spinner("Loading model..."):
            model = load_model(selected_checkpoint)
            if model is not None:
                st.session_state.model = model
                st.session_state.model_path = selected_checkpoint
                st.sidebar.success("Model loaded successfully!")
            else:
                st.sidebar.error("Failed to load model.")
    
    # Check if model is loaded
    if "model" not in st.session_state:
        st.info("👈 Please select and load a model from the sidebar.")
        st.stop()
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Input Image")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["png", "jpg", "jpeg"],
            help="Upload an image to analyze for anomalies"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Predict button
            if st.button("🔍 Detect Anomalies", type="primary", use_container_width=True):
                with st.spinner("Processing image..."):
                    # Preprocess
                    image_tensor, original_image = preprocess_image(image)
                    
                    # Predict
                    pred_score, pred_label, anomaly_map = predict_image(
                        st.session_state.model,
                        image_tensor
                    )
                    
                    # Store results
                    st.session_state.pred_score = pred_score
                    st.session_state.pred_label = pred_label
                    st.session_state.anomaly_map = anomaly_map
                    st.session_state.original_image = original_image
    
    with col2:
        st.header("Results")
        
        if "pred_score" in st.session_state:
            pred_score = st.session_state.pred_score
            pred_label = st.session_state.pred_label
            anomaly_map = st.session_state.anomaly_map
            original_image = st.session_state.original_image
            
            # Display score and label
            st.metric("Anomaly Score", f"{pred_score:.4f}" if pred_score is not None else "N/A")
            
            if pred_label is not None:
                status = "⚠️ Anomaly Detected" if pred_label == 1 else "✅ Normal"
                st.markdown(f"**Status:** {status}")
            
            # Display visualizations
            if anomaly_map is not None and original_image is not None:
                # Normalize and create heatmap
                anomaly_map_norm = normalize_map(anomaly_map)
                heatmap = to_heatmap(anomaly_map_norm)
                overlay = overlay_heatmap(original_image, heatmap)
                
                # Display heatmap
                st.subheader("Anomaly Heatmap")
                st.image(heatmap, use_container_width=True)
                
                # Display overlay
                st.subheader("Overlay")
                st.image(overlay, use_container_width=True)
            else:
                st.info("Anomaly map not available. Score only prediction.")
        else:
            st.info("👈 Upload an image and click 'Detect Anomalies' to see results.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        **Model Info:**
        - Model: PatchCore
        - Framework: Anomalib
        - Checkpoint: `{}`
        """.format(Path(st.session_state.get("model_path", "N/A")).relative_to(Path(".")) if "model_path" in st.session_state else "N/A")
    )


if __name__ == "__main__":
    main()
