
import numpy as np
import cupy as cp
import time

# Matrix size (adjust to your GPU memory, 4000 is usually safe for GT 740M)
N = 4000

# ---------------- CPU Benchmark ----------------
print(f"Generating {N}x{N} matrices on CPU...")
A_cpu = np.random.rand(N, N).astype(np.float32)
B_cpu = np.random.rand(N, N).astype(np.float32)

start = time.time()
C_cpu = A_cpu @ B_cpu  # CPU matrix multiplication
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.4f} seconds")

# ---------------- GPU Benchmark ----------------
print(f"Generating {N}x{N} matrices on GPU...")
A_gpu = cp.random.rand(N, N, dtype=cp.float32)
B_gpu = cp.random.rand(N, N, dtype=cp.float32)

cp.cuda.Stream.null.synchronize()  # make sure previous ops are done
start = time.time()
C_gpu = A_gpu @ B_gpu  # GPU matrix multiplication
cp.cuda.Stream.null.synchronize()  # wait for GPU to finish
gpu_time = time.time() - start
print(f"GPU time: {gpu_time:.4f} seconds")

# ---------------- Verification ----------------
C_gpu_cpu = cp.asnumpy(C_gpu)  # copy back to CPU to compare
print("Results match:", np.allclose(C_cpu, C_gpu_cpu, atol=1e-4))
