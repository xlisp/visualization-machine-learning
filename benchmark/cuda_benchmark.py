# cuda_benchmark.py
import torch
import time

# Set up the size of the tensor
n = 10000

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

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
    torch.cuda.synchronize() if device == 'cuda' else None  # Only for CUDA
    end = time.time()
    
    return end - start

# CPU Benchmark
cpu_time = benchmark('cpu', A, B)
print(f'CPU Time: {cpu_time:.4f} seconds')

# CUDA Benchmark
if cuda_available:
    cuda_time = benchmark('cuda', A, B)
    print(f'CUDA Time: {cuda_time:.4f} seconds')
else:
    print("CUDA not available")

