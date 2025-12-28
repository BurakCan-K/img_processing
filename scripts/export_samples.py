# Export samples script
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_default_data_root, ensure_dir, now_run_id, resolve_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export samples")
    
    parser.add_argument("--data_root", type=str, default=get_default_data_root(),
                        help="Root directory of the dataset")
    parser.add_argument("--category", type=str, required=True,
                        help="Category name (e.g., 'bottle', 'cable')")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size for export")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device to use (auto/cpu/cuda)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to export")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for exported samples")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Resolve device
    device = resolve_device(args.device)
    
    # Create output directory
    run_id = now_run_id()
    output_path = Path(args.output_dir) / "export_samples" / args.category / run_id
    ensure_dir(str(output_path))
    
    # Print arguments
    print("=" * 60)
    print("Export Samples Configuration")
    print("=" * 60)
    print(f"Data root:      {args.data_root}")
    print(f"Category:       {args.category}")
    print(f"Image size:     {args.image_size}")
    print(f"Device:         {device}")
    print(f"Model path:     {args.model_path}")
    print(f"Num samples:    {args.num_samples}")
    print(f"Output dir:     {output_path}")
    print("=" * 60)
    
    # TODO: Add export logic here


if __name__ == "__main__":
    main()
