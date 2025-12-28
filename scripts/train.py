# Training script
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
        # Alternative import path for older/newer versions
        from anomalib.datasets import MVTecAD
    except ImportError:
        raise ImportError("Could not import MVTecAD from anomalib.data or anomalib.datasets")

try:
    from anomalib.models import Patchcore
except ImportError:
    try:
        # Alternative import path
        from anomalib.models.patchcore import Patchcore
    except ImportError:
        raise ImportError("Could not import Patchcore from anomalib.models")

try:
    from anomalib.engine import Engine
except ImportError:
    # Alternative: use pytorch_lightning Trainer
    try:
        from pytorch_lightning import Trainer
        Engine = None  # Will use Trainer instead
    except ImportError:
        raise ImportError("Could not import Engine from anomalib.engine or pytorch_lightning.Trainer")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PatchCore model")
    
    parser.add_argument("--data_root", type=str, default=get_default_data_root(),
                        help="Root directory of the dataset")
    parser.add_argument("--category", type=str, default="carpet",
                        help="Category name (e.g., 'bottle', 'cable', 'carpet')")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size for training")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for training results")
    
    return parser.parse_args()


def main():
    import os
    import glob
    from pathlib import Path as PathBase
    # Disable symlink warnings on Windows
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    # Monkey patch Path.symlink_to to avoid Windows symlink issues
    # This prevents symlink errors on Windows without admin permissions
    try:
        from pathlib import Path as PathBase
        original_symlink_to = PathBase.symlink_to
        
        def patched_symlink_to(self, target, target_is_directory=False):
            """Patched symlink_to that silently fails on Windows instead of raising error."""
            try:
                return original_symlink_to(self, target, target_is_directory=target_is_directory)
            except (OSError, NotImplementedError):
                # On Windows without symlink permissions, skip symlink creation
                # The directory structure will still work without the symlink
                pass
        
        PathBase.symlink_to = patched_symlink_to
    except Exception:
        pass  # If patching fails, continue anyway
    
    args = parse_args()
    
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
    
    # Resolve device for PyTorch Lightning
    device_resolved = resolve_device(args.device)
    
    # PyTorch Lightning uses "gpu" instead of "cuda", or None for auto
    cuda_unavailable = False
    if device_resolved == "cuda":
        # Check if CUDA is actually available
        try:
            import torch
            if torch.cuda.is_available():
                device = "gpu"  # PyTorch Lightning uses "gpu"
            else:
                device = "cpu"
                cuda_unavailable = True
        except ImportError:
            device = "cpu"
            cuda_unavailable = True
    else:
        device = device_resolved  # "cpu" or "auto" -> None for Lightning
    
    # Create output directories
    run_id = now_run_id()
    model_dir = Path(args.output_dir) / "models" / run_id
    metrics_dir = Path(args.output_dir) / "metrics" / run_id
    ensure_dir(str(model_dir))
    ensure_dir(str(metrics_dir))
    
    # Print arguments
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Data root:      {args.data_root}")
    print(f"Category:       {args.category}")
    print(f"Image size:     {args.image_size}")
    # Show device info
    device_display = "cuda" if device_resolved == "cuda" and device == "gpu" else device_resolved
    if cuda_unavailable:
        device_display = "cuda (fallback to cpu)"
    
    print(f"Device:         {device_display}")
    if device == "cpu" or cuda_unavailable:
        print("⚠️  Warning: Using CPU. Training will be slow.")
        if cuda_unavailable:
            print("   CUDA was requested but is not available.")
        else:
            print("   If you have a GPU, use --device cuda for faster training.")
    print(f"Model dir:      {model_dir}")
    print(f"Metrics dir:    {metrics_dir}")
    print("=" * 60)
    
    # Create datamodule (try different API versions)
    print("\nCreating datamodule...")
    try:
        # Try standard API (without image_size - handled by transforms/model)
        datamodule = MVTecAD(
            root=args.data_root,
            category=args.category,
        )
    except TypeError:
        # Alternative API: try with different parameter names
        try:
            datamodule = MVTecAD(
                data_path=args.data_root,
                category=args.category,
            )
        except TypeError:
            # Last fallback: try with all common parameters
            datamodule = MVTecAD(
                root=str(args.data_root),
                category=args.category,
                train_batch_size=32,
                eval_batch_size=32,
            )
    
    # Create model
    print("Creating PatchCore model...")
    print("Note: First run will download pre-trained weights (~100-200 MB).")
    print("      This is a one-time download and will be cached for future use.")
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
    )
    
    # Create engine/trainer
    print("Setting up training engine...")
    # Use Anomalib Engine with proper accelerator setting
    if Engine is not None:
        # Convert PyTorch Lightning device format to Engine format
        # device is "gpu" or "cpu" after resolution
        # Engine uses "cuda" for GPU, so convert "gpu" -> "cuda"
        if device == "gpu":
            accelerator_for_engine = "cuda"
        elif device == "cpu":
            accelerator_for_engine = "cpu"
        else:
            accelerator_for_engine = None  # auto
        
        try:
            engine = Engine(
                max_epochs=1,  # PatchCore typically needs 1 epoch
                accelerator=accelerator_for_engine,
                default_root_dir=str(model_dir),
            )
        except Exception as e:
            print(f"Error: Engine initialization failed: {e}")
            raise
    else:
        # Fallback to pytorch_lightning Trainer
        from pytorch_lightning import Trainer
        accelerator = device if device != "auto" else None
        if accelerator == "cuda":
            import torch
            if not torch.cuda.is_available():
                accelerator = "cpu"
        engine = Trainer(
            max_epochs=1,
            accelerator=accelerator,
            default_root_dir=str(model_dir),
        )
    
    # Train
    print("Starting training...")
    engine.fit(model=model, datamodule=datamodule)
    
    # Find the checkpoint file created by Engine/Trainer
    # Engine creates: default_root_dir/Patchcore/MVTecAD/category/v0/...
    # Trainer creates: default_root_dir/...
    model_path = None
    
    if Engine is not None:
        # Engine creates its own directory structure
        # Search for checkpoint files recursively
        checkpoint_pattern = str(model_dir / "**" / "*.ckpt")
        checkpoint_files = glob.glob(checkpoint_pattern, recursive=True)
        if checkpoint_files:
            # Use the most recent checkpoint
            model_path = Path(max(checkpoint_files, key=os.path.getmtime))
            print(f"Checkpoint saved at: {model_path}")
        else:
            # Also search in parent directory (Engine might create structure there)
            parent_checkpoint_pattern = str(model_dir.parent / "**" / "*.ckpt")
            checkpoint_files = glob.glob(parent_checkpoint_pattern, recursive=True)
            if checkpoint_files:
                model_path = Path(max(checkpoint_files, key=os.path.getmtime))
                print(f"Checkpoint saved at: {model_path}")
    else:
        # Trainer saves to default_root_dir
        checkpoint_files = list(model_dir.glob("*.ckpt"))
        if checkpoint_files:
            model_path = checkpoint_files[0]
            preferred = model_dir / "model.ckpt"
            if preferred.exists():
                model_path = preferred
            print(f"Checkpoint saved at: {model_path}")
    
    # If no checkpoint found, try to save manually
    if model_path is None:
        model_path = model_dir / "model.ckpt"
        print(f"No checkpoint found, saving manually to {model_path}...")
        try:
            import torch
            torch.save(model.state_dict(), str(model_path))
        except Exception as e:
            print(f"Warning: Could not save model: {e}")
            model_path = None
    
    # Save training config
    train_config = {
        "data_root": args.data_root,
        "category": args.category,
        "image_size": args.image_size,
        "device": device,
        "run_id": run_id,
        "model_path": str(model_path) if model_path else None,
    }
    config_path = metrics_dir / "train_config.json"
    print(f"Saving training config to {config_path}...")
    with open(config_path, "w") as f:
        json.dump(train_config, f, indent=2)
    
    print("\nTraining completed successfully!")
    if model_path:
        print(f"Model saved to: {model_path}")
    else:
        print("Warning: Model checkpoint location could not be determined.")
        print(f"Check Engine output directory: {Path(args.output_dir) / 'models'}")
    print(f"Config saved to: {config_path}")


if __name__ == "__main__":
    main()
