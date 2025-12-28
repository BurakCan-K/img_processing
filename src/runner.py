# Runner module

def main():
    """Sanity check for dependencies and CUDA availability."""
    print("Running sanity check...")
    
    # Check anomalib import
    try:
        import anomalib
        print(f"✓ anomalib imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import anomalib: {e}")
        return
    
    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"✓ torch imported successfully")
        print(f"CUDA available: {cuda_available}")
        if cuda_available:
            print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ Failed to import torch: {e}")
        return
    
    print("Sanity check completed!")


if __name__ == "__main__":
    main()
