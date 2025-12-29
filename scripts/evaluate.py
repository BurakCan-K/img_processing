# Evaluation script
import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_default_data_root, ensure_dir, now_run_id, resolve_device

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

try:
    from anomalib.engine import Engine
except ImportError:
    Engine = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate PatchCore model")
    
    parser.add_argument("--data_root", type=str, default=get_default_data_root(),
                        help="Root directory of the dataset")
    parser.add_argument("--category", type=str, required=True,
                        help="Category name (e.g., 'bottle', 'cable', 'carpet')")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size for evaluation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for evaluation results")
    
    return parser.parse_args()


def main():
    import numpy as np
    from sklearn.metrics import roc_auc_score
    import torch
    
    args = parse_args()
    
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
    cuda_unavailable = False
    
    if device_resolved == "cuda":
        try:
            if torch.cuda.is_available():
                accelerator_for_engine = "cuda"
            else:
                accelerator_for_engine = "cpu"
                cuda_unavailable = True
        except ImportError:
            accelerator_for_engine = "cpu"
            cuda_unavailable = True
    else:
        accelerator_for_engine = "cpu" if device_resolved == "cpu" else None
    
    # Create output directories
    run_id = now_run_id()
    metrics_dir = Path(args.output_dir) / "metrics" / run_id
    ensure_dir(str(metrics_dir))
    
    # Print arguments
    print("=" * 60)
    print("Evaluation Configuration")
    print("=" * 60)
    print(f"Data root:      {args.data_root}")
    print(f"Category:       {args.category}")
    print(f"Image size:     {args.image_size}")
    device_display = accelerator_for_engine if accelerator_for_engine else "auto"
    if cuda_unavailable:
        device_display = "cuda (fallback to cpu)"
    print(f"Device:         {device_display}")
    print(f"Model path:     {args.model_path}")
    print(f"Metrics dir:    {metrics_dir}")
    print("=" * 60)
    
    # Create datamodule for test
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
        model = Patchcore.load_from_checkpoint(str(model_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load model from checkpoint: {e}")
    
    # Try to get metrics from Engine.test() first
    image_auroc = None
    n_test = None
    
    print("\nRunning evaluation on test set...")
    if Engine is not None:
        try:
            engine = Engine(accelerator=accelerator_for_engine)
            test_results = engine.test(model=model, datamodule=datamodule, verbose=False)
            
            if test_results and len(test_results) > 0:
                result_dict = test_results[0]
                
                # Look for image-level AUROC
                for key in result_dict.keys():
                    key_lower = key.lower()
                    if "image" in key_lower and "auroc" in key_lower:
                        image_auroc = float(result_dict[key])
                        print(f"✓ Found image-level AUROC in test results: {key} = {image_auroc:.4f}")
                        break
                
                # Try to find test sample count
                for key in result_dict.keys():
                    key_lower = key.lower()
                    if "test" in key_lower and any(x in key_lower for x in ["size", "count", "n", "samples"]):
                        n_test = int(result_dict[key])
                        break
        except Exception as e:
            print(f"Warning: Engine.test() failed: {e}")
            test_results = None
    else:
        test_results = None
    
    # If AUROC not found, calculate manually
    if image_auroc is None:
        print("Calculating AUROC manually from predictions...")
        
        from pytorch_lightning import Trainer
        
        # Setup datamodule
        datamodule.setup("test")
        test_dataloader = datamodule.test_dataloader()
        
        # Collect predictions and labels
        all_scores = []
        all_labels = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader):
                # Extract labels from batch
                if isinstance(batch, dict):
                    labels = batch.get("label", None)
                elif hasattr(batch, "label"):
                    labels = batch.label
                else:
                    labels = None
                
                # Get predictions using model's test_step or predict_step
                try:
                    # Use test_step which is designed for evaluation
                    output = model.test_step(batch, batch_idx)
                except Exception:
                    try:
                        # Fallback to predict_step
                        output = model.predict_step(batch, batch_idx)
                    except Exception:
                        try:
                            # Fallback to validation_step
                            output = model.validation_step(batch, batch_idx)
                        except Exception as e:
                            print(f"Warning: Could not get predictions for batch {batch_idx}: {e}")
                            continue
                
                # Extract anomaly scores
                if isinstance(output, dict):
                    # Try different keys for anomaly score
                    score = None
                    for key in ["pred_score", "anomaly_score", "score", "pred_scores", "image_pred_score"]:
                        if key in output:
                            score = output[key]
                            break
                    
                    if score is None:
                        # Take first tensor value
                        for val in output.values():
                            if torch.is_tensor(val):
                                score = val
                                break
                    
                    if score is not None:
                        if torch.is_tensor(score):
                            score = score.cpu().numpy()
                        
                        # Flatten to image-level (mean if multi-dimensional)
                        if score.ndim > 1:
                            score = score.reshape(score.shape[0], -1).mean(axis=1)
                        
                        all_scores.extend(score.flatten())
                
                # Extract labels if available
                if labels is not None:
                    if torch.is_tensor(labels):
                        labels = labels.cpu().numpy()
                    
                    # Convert to binary: 0=normal, 1=anomaly
                    if labels.ndim > 1:
                        # For masks, check if any pixel is anomalous
                        labels = labels.reshape(labels.shape[0], -1).max(axis=1)
                    
                    all_labels.extend(labels.flatten())
        
        # Calculate AUROC
        if all_scores and all_labels and len(all_scores) == len(all_labels):
            all_scores = np.array(all_scores)
            all_labels = np.array(all_labels)
            
            try:
                image_auroc = roc_auc_score(all_labels, all_scores)
                n_test = len(all_scores)
                print(f"✓ Calculated image-level AUROC: {image_auroc:.4f} (from {n_test} samples)")
            except Exception as e:
                print(f"Warning: Could not calculate AUROC: {e}")
        else:
            print(f"Warning: Could not collect predictions. Scores: {len(all_scores) if all_scores else 0}, Labels: {len(all_labels) if all_labels else 0}")
    
    # Prepare metrics
    metrics = {
        "category": args.category,
        "n_test": int(n_test) if n_test is not None else 0,
        "device": accelerator_for_engine if accelerator_for_engine else "auto",
        "image_auroc": float(image_auroc) if image_auroc is not None else None,
    }
    
    # Save metrics
    metrics_path = metrics_dir / "metrics.json"
    print(f"\nSaving metrics to {metrics_path}...")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Category:       {metrics['category']}")
    print(f"Test samples:   {metrics['n_test']}")
    print(f"Device:         {metrics['device']}")
    if metrics['image_auroc'] is not None:
        print(f"Image AUROC:    {metrics['image_auroc']:.4f}")
    else:
        print(f"Image AUROC:    N/A")
    print("=" * 60)
    print(f"\nMetrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
