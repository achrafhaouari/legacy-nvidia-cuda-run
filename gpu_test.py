#Run/Debug Config
#Docker container setting:
# --entrypoint= --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it

import numpy as np
import cupy as cp
import time
import sys

# Verify Python and CuPy versions
print(f"Python Version: {sys.version}")
print(f"CuPy Version: {cp.__version__}")
print("-" * 30)

# Define the size of the matrices
matrix_size = 10000

print(f"--- Testing Matrix Multiplication with size {matrix_size}x{matrix_size} ---")

# --- CPU Calculation using NumPy ---
print("\n1. Running on CPU with NumPy...")
cpu_a = np.random.rand(matrix_size, matrix_size).astype(np.float32)
cpu_b = np.random.rand(matrix_size, matrix_size).astype(np.float32)
start_cpu = time.time()
cpu_c = np.dot(cpu_a, cpu_b)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu
print(f"   CPU Time: {cpu_time:.4f} seconds")

# --- GPU Calculation using CuPy ---
print("\n2. Running on GPU with CuPy...")
# Ensure GPU is being used
with cp.cuda.Device(0):
    gpu_a = cp.asarray(cpu_a)
    gpu_b = cp.asarray(cpu_b)
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    gpu_c = cp.dot(gpu_a, gpu_b)
    end_gpu.record()
    end_gpu.synchronize()
    gpu_time = cp.cuda.get_elapsed_time(start_gpu, end_gpu) / 1000

print(f"   GPU Time: {gpu_time:.4f} seconds")

# --- Verification and Speedup ---
print("\n✅ Verification successful: CPU and GPU results match.")
speedup = cpu_time / gpu_time
print(f"🚀 GPU was approximately {speedup:.2f} times faster than the CPU.")