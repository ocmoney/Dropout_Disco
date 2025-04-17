import torch
import sys
import platform

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"Platform: {platform.platform()}")

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    # Create a test tensor on GPU
    x = torch.rand(5, 3).cuda()
    print(f"Test tensor on GPU: {x}")
    print(f"Tensor device: {x.device}")
else:
    print("No CUDA devices available. Using CPU.")
    
    # Create a test tensor on CPU
    x = torch.rand(5, 3)
    print(f"Test tensor on CPU: {x}")
    print(f"Tensor device: {x.device}")

# Check MPS (Apple Silicon) availability
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print(f"MPS (Apple Silicon) available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    
    # Create a test tensor on MPS
    x = torch.rand(5, 3).to(torch.device("mps"))
    print(f"Test tensor on MPS: {x}")
    print(f"Tensor device: {x.device}")
else:
    print("MPS (Apple Silicon) not available.")

print("\nDevice check complete.") 