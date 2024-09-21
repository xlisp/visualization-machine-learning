# mps_benchmark.py
import torch
import time

# Set up the size of the tensor
n = 10000

# Check if MPS is available (Apple Silicon)
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

# Generate random tensors
A = torch.randn(n, n)
B = torch.randn(n, n)

# Function to measure time
def benchmark(device, A, B):
    A, B = A.to(device), B.to(device)
    
    # Warm up
    C = A @ B
    
    # Measure the time
    start = time.time()
    C = A @ B
    end = time.time()
    
    return end - start

# CPU Benchmark
cpu_time = benchmark('cpu', A, B)
print(f'CPU Time: {cpu_time:.4f} seconds')

# MPS Benchmark
if mps_available:
    mps_time = benchmark('mps', A, B)
    print(f'MPS Time: {mps_time:.4f} seconds')
else:
    print("MPS not available")

