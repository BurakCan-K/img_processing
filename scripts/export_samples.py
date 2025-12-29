# Export samples script
import argparse
import sys
import random
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_default_data_root, ensure_dir, now_run_id, resolve_device
from src.viz import save_triplet

# Try different import paths for anomalib (API may vary between versions)
try:
    from anomalib.data import MVTecAD
except ImportError:
    try:
        from anomalib.datasets import MVTecAD
    except ImportError:
        raise ImportError("Could not import MVTecAD from anomalib.data or anomalib.datasets")

try:
    from anomalib.models import Patchcore
except ImportError:
    try:
        from anomalib.models.patchcore import Patchcore
    except ImportError:
        raise ImportError("Could not import Patchcore from anomalib.models")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export visualization samples")
    
    parser.add_argument("--data_root", type=str, default=get_default_data_root(),
                        help="Root directory of the dataset")
    parser.add_argument("--category", type=str, required=True,
                        help="Category name (e.g., 'bottle', 'cable', 'carpet')")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size for export")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to export")
    parser.add_argument("--random", action="store_true",
                        help="Select random samples instead of first N")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample selection")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for exported samples")
    
    return parser.parse_args()


def main():
    import torch
    import numpy as np
    from PIL import Image
    
    args = parse_args()
    
    # Set random seed if using random selection
    if args.random:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Check if model path exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {args.model_path}\n"
            f"Please ensure the model path is correct."
        )
    
    # Check if dataset path exists
    data_root_path = Path(args.data_root)
    if not data_root_path.exists():
        raise FileNotFoundError(
            f"Dataset root directory not found: {args.data_root}\n"
            f"Please ensure the MVTecAD dataset is downloaded and extracted to the correct location."
        )
    
    category_path = data_root_path / args.category
    if not category_path.exists():
        raise FileNotFoundError(
            f"Category directory not found: {category_path}\n"
            f"Available categories in {data_root_path}: {list(data_root_path.iterdir()) if data_root_path.exists() else 'N/A'}"
        )
    
    # Resolve device
    device_resolved = resolve_device(args.device)
    if device_resolved == "cuda":
        try:
            if torch.cuda.is_available():
                accelerator = "cuda"
            else:
                accelerator = "cpu"
        except ImportError:
            accelerator = "cpu"
    else:
        accelerator = "cpu" if device_resolved == "cpu" else None
    
    # Create output directory
    run_id = now_run_id()
    output_path = Path(args.output_dir) / "samples" / run_id
    ensure_dir(str(output_path))
    
    # Print arguments
    print("=" * 60)
    print("Export Samples Configuration")
    print("=" * 60)
    print(f"Data root:      {args.data_root}")
    print(f"Category:       {args.category}")
    print(f"Image size:     {args.image_size}")
    print(f"Device:         {accelerator if accelerator else 'auto'}")
    print(f"Model path:     {args.model_path}")
    print(f"Num samples:    {args.num_samples}")
    print(f"Selection:      {'random' if args.random else 'first N'}")
    if args.random:
        print(f"Random seed:    {args.seed}")
    print(f"Output dir:     {output_path}")
    print("=" * 60)
    
    # Create datamodule
    print("\nCreating datamodule...")
    try:
        datamodule = MVTecAD(
            root=args.data_root,
            category=args.category,
        )
    except TypeError:
        try:
            datamodule = MVTecAD(
                data_path=args.data_root,
                category=args.category,
            )
        except TypeError:
            datamodule = MVTecAD(
                root=str(args.data_root),
                category=args.category,
                train_batch_size=32,
                eval_batch_size=32,
            )
    
    # Load model from checkpoint
    print(f"\nLoading model from {args.model_path}...")
    try:
        import torch
        model = Patchcore.load_from_checkpoint(str(model_path))
        model.eval() # Set model to evaluation mode
        
        # Move model to device
        if accelerator == "cuda" and torch.cuda.is_available():
            model = model.cuda()
        else:
            model = model.cpu()
    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")
    
    # Setup datamodule and get test dataloader
    print("Setting up test dataloader...")
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()
    
    # Collect all samples (images and their indices)
    print("Collecting test samples...")
    all_samples = []
    sample_idx = 0
    
    for batch in test_dataloader:
        # Check batch type and extract images
        if isinstance(batch, dict):
            images = batch.get("image", None)
        elif isinstance(batch, (list, tuple)):
            images = batch[0]
        elif hasattr(batch, "image"):
            images = batch.image
        else:
            continue
        
        if images is None:
            continue
        
        # Determine batch type for later reconstruction
        batch_type = type(batch)
        
        batch_size = images.shape[0]
        for i in range(batch_size):
            all_samples.append({
                "batch": batch,  # Keep original batch
                "batch_type": batch_type,  # Store type for reconstruction
                "batch_idx": i,
                "sample_idx": sample_idx
            })
            sample_idx += 1
    
    total_samples = len(all_samples)
    print(f"Total test samples available: {total_samples}")
    
    # Select samples
    num_to_export = min(args.num_samples, total_samples)
    if args.random:
        selected_samples = random.sample(all_samples, num_to_export)
    else:
        selected_samples = all_samples[:num_to_export]
    
    # Export samples
    print(f"\nExporting {num_to_export} samples...")
    model.eval()
    
    exported_count = 0
    
    with torch.no_grad():
        for sample_data in selected_samples:
            original_batch = sample_data["batch"]
            batch_type = sample_data.get("batch_type", type(original_batch))
            batch_idx_in_batch = sample_data["batch_idx"]
            sample_idx = sample_data["sample_idx"]
            
            # Create single-item batch from original batch format
            # Check if original batch is already ImageBatch (hasattr check first)
            if hasattr(original_batch, "image"):
                # ImageBatch or similar object - extract single item
                original_image = original_batch.image
                # Slice the tensor
                if torch.is_tensor(original_image):
                    single_image = original_image[batch_idx_in_batch:batch_idx_in_batch+1]
                else:
                    single_image = original_image[batch_idx_in_batch:batch_idx_in_batch+1]
                
                if accelerator == "cuda" and torch.cuda.is_available():
                    single_image = single_image.cuda()
                
                # Create new ImageBatch with same type as original
                try:
                    single_batch = batch_type(image=single_image)
                except Exception:
                    # Fallback: try ImageBatch import
                    try:
                        from anomalib.data.utils import ImageBatch
                        single_batch = ImageBatch(image=single_image)
                    except:
                        # Last fallback: use dict
                        single_batch = {"image": single_image}
            elif isinstance(original_batch, dict):
                # Extract single image from batch
                if "image" in original_batch:
                    single_image = original_batch["image"][batch_idx_in_batch:batch_idx_in_batch+1]
                else:
                    # Try to find image tensor
                    for key, value in original_batch.items():
                        if torch.is_tensor(value) and len(value.shape) >= 3:
                            single_image = value[batch_idx_in_batch:batch_idx_in_batch+1]
                            break
                    else:
                        print(f"Warning: Could not find image in batch for sample {sample_idx}")
                        continue
                
                # Move to device
                if accelerator == "cuda" and torch.cuda.is_available():
                    single_image = single_image.cuda()
                
                # Convert to ImageBatch format (model expects this)
                try:
                    from anomalib.data.utils import ImageBatch
                    single_batch = ImageBatch(image=single_image)
                    # Debug for first sample
                    if sample_idx == 0:
                        print(f"Debug: Created ImageBatch, type={type(single_batch)}, has image={hasattr(single_batch, 'image')}")
                except ImportError as e:
                    # Fallback: use dict but ensure "image" key exists
                    if sample_idx == 0:
                        print(f"Debug: ImageBatch import failed: {e}, using dict fallback")
                    single_batch = {"image": single_image}
                    # Copy other keys if needed
                    for key, value in original_batch.items():
                        if key != "image" and torch.is_tensor(value):
                            single_value = value[batch_idx_in_batch:batch_idx_in_batch+1]
                            if accelerator == "cuda" and torch.cuda.is_available():
                                single_value = single_value.cuda()
                            single_batch[key] = single_value
            else:
                # Fallback: try to extract image from batch
                if isinstance(original_batch, (list, tuple)):
                    images = original_batch[0]
                else:
                    images = original_batch
                
                # If images is an ImageBatch or similar object, get the tensor first
                if hasattr(images, "image"):
                    images = images.image
                
                # Now slice the tensor
                if torch.is_tensor(images):
                    single_image = images[batch_idx_in_batch:batch_idx_in_batch+1]
                else:
                    print(f"Warning: Could not extract image tensor for sample {sample_idx}")
                    continue
                
                if accelerator == "cuda" and torch.cuda.is_available():
                    single_image = single_image.cuda()
                
                # Try to create ImageBatch
                try:
                    from anomalib.data.utils import ImageBatch
                    single_batch = ImageBatch(image=single_image)
                except:
                    single_batch = {"image": single_image}
            
            # Get prediction using model methods directly (bypass Trainer)
            # Use same approach as evaluate.py - pass batch directly to test_step
            try:
                # Try test_step first (works in evaluate.py)
                # Note: test_step expects the batch format from dataloader
                output = model.test_step(single_batch, batch_idx=0)
            except Exception as e1:
                # If test_step fails, try to understand the error
                error_msg = str(e1)
                if "'dict' object has no attribute 'image'" in error_msg:
                    # Model expects ImageBatch or different format
                    # Try converting dict to ImageBatch
                    try:
                        from anomalib.data.utils import ImageBatch
                        if isinstance(single_batch, dict) and "image" in single_batch:
                            single_batch = ImageBatch(image=single_batch["image"])
                            output = model.test_step(single_batch, batch_idx=0)
                        else:
                            raise e1
                    except Exception as e1b:
                        try:
                            # Fallback to predict_step
                            output = model.predict_step(single_batch, batch_idx=0)
                        except Exception as e2:
                            try:
                                # Fallback to validation_step
                                output = model.validation_step(single_batch, batch_idx=0)
                            except Exception as e3:
                                try:
                                    # Last resort: use forward directly
                                    output = model(single_batch)
                                except Exception as e4:
                                    print(f"Warning: Could not get prediction for sample {sample_idx}")
                                    print(f"  test_step: {str(e1)[:100]}")
                                    print(f"  predict_step: {str(e2)[:100]}")
                                    print(f"  validation_step: {str(e3)[:100]}")
                                    print(f"  forward: {str(e4)[:100]}")
                                    continue
                else:
                    # Other error, try fallbacks
                    try:
                        output = model.predict_step(single_batch, batch_idx=0)
                    except Exception as e2:
                        try:
                            output = model.validation_step(single_batch, batch_idx=0)
                        except Exception as e3:
                            try:
                                output = model(single_batch)
                            except Exception as e4:
                                print(f"Warning: Could not get prediction for sample {sample_idx}")
                                print(f"  test_step: {str(e1)[:100]}")
                                print(f"  predict_step: {str(e2)[:100]}")
                                print(f"  validation_step: {str(e3)[:100]}")
                                print(f"  forward: {str(e4)[:100]}")
                                continue
            
            # Extract pred_score, pred_label, and anomaly_map
            pred_score = None
            pred_label = None
            anomaly_map = None
            
            # Handle ImageBatch output
            if hasattr(output, "pred_score") or hasattr(output, "anomaly_score"):
                # ImageBatch output - extract attributes
                if hasattr(output, "pred_score"):
                    pred_score = output.pred_score
                elif hasattr(output, "anomaly_score"):
                    pred_score = output.anomaly_score
                
                if hasattr(output, "pred_label"):
                    pred_label = output.pred_label
                
                if hasattr(output, "anomaly_map"):
                    anomaly_map = output.anomaly_map
                
                # Convert tensors to numpy
                if torch.is_tensor(pred_score):
                    pred_score = pred_score.cpu().item() if pred_score.numel() == 1 else pred_score.cpu().numpy()[0]
                if torch.is_tensor(pred_label):
                    pred_label = int(pred_label.cpu().item() if pred_label.numel() == 1 else pred_label.cpu().numpy()[0])
                if torch.is_tensor(anomaly_map):
                    anomaly_map = anomaly_map.cpu().numpy()
                    if anomaly_map.ndim > 2 and anomaly_map.shape[0] == 1:
                        anomaly_map = anomaly_map[0]
            
            elif isinstance(output, dict):
                # Try to get pred_score
                for key in ["pred_score", "anomaly_score", "score", "image_pred_score"]:
                    if key in output:
                        pred_score = output[key]
                        if torch.is_tensor(pred_score):
                            pred_score = pred_score.cpu().item() if pred_score.numel() == 1 else pred_score.cpu().numpy()
                            if isinstance(pred_score, np.ndarray):
                                pred_score = pred_score.item() if pred_score.size == 1 else pred_score[0]
                        break
                
                # Try to get pred_label
                for key in ["pred_label", "label", "pred_class"]:
                    if key in output:
                        pred_label = output[key]
                        if torch.is_tensor(pred_label):
                            pred_label = int(pred_label.cpu().item() if pred_label.numel() == 1 else pred_label.cpu().numpy())
                            if isinstance(pred_label, np.ndarray):
                                pred_label = int(pred_label.item() if pred_label.size == 1 else pred_label[0])
                        break
                
                # Infer pred_label from score if not found
                if pred_label is None and pred_score is not None:
                    pred_label = 1 if pred_score > 0.5 else 0
                
                # Try to get anomaly_map
                for key in ["anomaly_map", "pred_mask", "mask", "heatmap", "anomaly_maps"]:
                    if key in output:
                        anomaly_map = output[key]
                        if torch.is_tensor(anomaly_map):
                            anomaly_map = anomaly_map.cpu().numpy()
                        # Remove batch dimension if present
                        if anomaly_map.ndim > 2 and anomaly_map.shape[0] == 1:
                            anomaly_map = anomaly_map[0]
                        break
                
                # Alternative: try model's internal state
                if anomaly_map is None:
                    for attr in ["anomaly_map", "anomaly_maps", "last_anomaly_map"]:
                        if hasattr(model, attr):
                            anomaly_map = getattr(model, attr)
                            if torch.is_tensor(anomaly_map):
                                anomaly_map = anomaly_map.cpu().numpy()
                            if anomaly_map.ndim > 2 and anomaly_map.shape[0] == 1:
                                anomaly_map = anomaly_map[0]
                            break
            
            # If still no anomaly_map, create from score
            if anomaly_map is None:
                if pred_score is not None:
                    # Get image dimensions from single_batch
                    if isinstance(single_batch, dict) and "image" in single_batch:
                        img = single_batch["image"]
                        if torch.is_tensor(img):
                            h, w = img.shape[-2:] if len(img.shape) > 2 else (256, 256)
                        else:
                            h, w = (256, 256)
                    elif hasattr(single_batch, "image"):
                        img = single_batch.image
                        if torch.is_tensor(img):
                            h, w = img.shape[-2:] if len(img.shape) > 2 else (256, 256)
                        else:
                            h, w = (256, 256)
                    else:
                        h, w = (256, 256)
                    anomaly_map = np.ones((h, w), dtype=np.float32) * float(pred_score)
                else:
                    print(f"Warning: Could not get anomaly_map or pred_score for sample {sample_idx}, skipping...")
                    if sample_idx == 0:
                        print(f"  Debug: output type={type(output)}, output keys={list(output.keys()) if isinstance(output, dict) else 'N/A'}")
                    continue
            
            # Ensure anomaly_map is 2D
            if anomaly_map.ndim > 2:
                anomaly_map = anomaly_map.squeeze()
            if anomaly_map.ndim == 0:
                anomaly_map = np.array([[anomaly_map]])
            elif anomaly_map.ndim == 1:
                # Reshape 1D to 2D (assume square)
                size = int(np.sqrt(anomaly_map.size))
                anomaly_map = anomaly_map.reshape(size, size)
            
            # Convert image to numpy for visualization
            # Get image from single_batch
            if isinstance(single_batch, dict) and "image" in single_batch:
                input_img = single_batch["image"][0]  # Remove batch dimension
            elif hasattr(single_batch, "image"):
                input_img = single_batch.image[0]
            else:
                input_img = single_batch[0] if isinstance(single_batch, (list, tuple)) else single_batch
            
            if torch.is_tensor(input_img):
                input_img = input_img.cpu().numpy()
            
            # Convert from CHW to HWC if needed
            if input_img.ndim == 3 and input_img.shape[0] in [1, 3]:
                input_img = np.transpose(input_img, (1, 2, 0))
            
            # Denormalize if needed (ImageNet normalization)
            if input_img.min() < -0.5 or input_img.max() > 2.0:
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                if input_img.ndim == 3 and input_img.shape[-1] == 3:
                    input_img = input_img * std + mean
                input_img = np.clip(input_img, 0, 1)
            
            # Convert to uint8
            if input_img.max() <= 1.0:
                input_img = (input_img * 255).astype(np.uint8)
            else:
                input_img = input_img.astype(np.uint8)
            
            # Ensure RGB (3 channels)
            if input_img.ndim == 2:
                input_img = np.stack([input_img] * 3, axis=-1)
            elif input_img.ndim == 3 and input_img.shape[-1] == 1:
                input_img = np.repeat(input_img, 3, axis=-1)
            
            # Prepare filename
            score_str = f"{pred_score:.3f}" if pred_score is not None else "0.000"
            label_str = f"{pred_label}" if pred_label is not None else "0"
            base_name = f"{sample_idx:04d}_label{label_str}_score{score_str}"
            
            # Save triplet
            try:
                save_triplet(output_path, base_name, input_img, anomaly_map)
                exported_count += 1
            except Exception as e:
                print(f"Warning: Could not save sample {sample_idx}: {e}")
                continue
    
    print("\n" + "=" * 60)
    print(f"Export completed: {exported_count} samples saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
